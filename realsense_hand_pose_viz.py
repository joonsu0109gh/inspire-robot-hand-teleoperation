#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time hand pose visualization using Intel RealSense and MediaPipe Hands.
Now includes:
- Per-finger curl estimation
- Curl value visualization on the 2D overlay
"""

import argparse
import time
from collections import deque

import numpy as np
import cv2
import pyrealsense2 as rs
import mediapipe as mp

# Optional Open3D
try:
    import open3d as o3d
    OPEN3D_OK = True
except Exception:
    OPEN3D_OK = False


# ============================================================
# Finger curl utilities
# ============================================================

# MediaPipe landmark indices per finger
FINGER_LANDMARKS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "little": [17, 18, 19, 20],
}

def compute_finger_curl(xyz, finger_name):
    """
    Compute curl value (0~1) for the given finger.
    Curl definition:
        - Use MCP→PIP vector and MCP→TIP vector.
        - Compute angle using normalized dot product.
        - dot = 1   → fully extended → curl = 0
        - dot = -1  → fully bent     → curl = 1
    """
    idx = FINGER_LANDMARKS[finger_name]
    mcp, pip, dip, tip = idx

    p_mcp = xyz[mcp]
    p_pip = xyz[pip]
    p_tip = xyz[tip]

    if not (np.all(np.isfinite(p_mcp)) and np.all(np.isfinite(p_pip)) and np.all(np.isfinite(p_tip))):
        return None

    v1 = p_pip - p_mcp
    v2 = p_tip - p_mcp

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    v1 /= n1
    v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))

    curl = 1.0 - (dot + 1.0) / 2.0
    return float(np.clip(curl, 0.0, 1.0))


def estimate_all_finger_curls(xyz):
    """Compute curl for all five fingers."""
    curls = {}
    for name in FINGER_LANDMARKS:
        curls[name] = compute_finger_curl(xyz, name)
    return curls


# ============================================================
# RealSense helpers
# ============================================================

def get_realsense_pipeline(width=1280, height=720, fps=30):
    """Start RealSense pipeline and return intrinsics."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    intrinsics = {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy,
        "width": intr.width,
        "height": intr.height,
        "depth_scale": depth_scale,
    }
    return pipeline, align, intrinsics


def median_depth_at(depth_img, u, v, k=5):
    """Return median depth around pixel (u,v) in a k×k window."""
    h, w = depth_img.shape
    u0 = max(0, u - k // 2)
    v0 = max(0, v - k // 2)
    u1 = min(w, u0 + k)
    v1 = min(h, v0 + k)

    window = depth_img[v0:v1, u0:u1]
    flat = window.flatten()
    flat = flat[flat > 0]
    if flat.size == 0:
        return None
    return np.median(flat)


def backproject(u, v, z, intr):
    """Convert pixel (u,v,z) to 3D camera coordinates."""
    X = (u - intr["cx"]) * z / intr["fx"]
    Y = (v - intr["cy"]) * z / intr["fy"]
    return np.array([X, Y, z], dtype=np.float32)


# ============================================================
# Open3D utilities
# ============================================================

def opend3d_init():
    """Initialize Open3D visualizer."""
    vis = o3d.visualization.Visualizer()
    vis.create_window("Hand 3D", width=800, height=600)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.2, 0.7, 1.0], (21, 1)))

    mp_hands = mp.solutions.hands
    lines = np.array(list(mp_hands.HAND_CONNECTIONS), dtype=np.int32)

    line_set = o3d.geometry.LineSet()
    line_set.points = pcd.points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile([1, 0.6, 0.2], (len(lines), 1)))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(axis)

    return vis, pcd, line_set


def opend3d_update(vis, pcd, line_set, xyz):
    """Update Open3D scene with new points."""
    xyz_valid = xyz.copy()
    xyz_valid[~np.isfinite(xyz_valid)] = 0.0
    pcd.points = o3d.utility.Vector3dVector(xyz_valid.astype(np.float64))
    line_set.points = pcd.points

    vis.update_geometry(pcd)
    vis.update_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()


# ============================================================
# Overlay drawing with curl visualization
# ============================================================

def draw_2d_overlay(frame, lm_px, handedness, curls=None, fps=None):
    """Draw 2D skeleton + curl text overlay."""
    mp_hands = mp.solutions.hands

    # Draw skeleton lines
    for a, b in mp_hands.HAND_CONNECTIONS:
        if lm_px[a] is None or lm_px[b] is None:
            continue
        cv2.line(frame, lm_px[a], lm_px[b], (0, 215, 255), 2)

    # Draw landmarks
    for idx, pt in enumerate(lm_px):
        if pt is None:
            continue
        cv2.circle(frame, pt, 4, (255, 170, 30), -1)

    # Basic info
    info = f"Hand: {handedness}"
    if fps is not None:
        info += f" | FPS: {fps:.1f}"
    cv2.putText(frame, info, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2)

    # Curl values
    if curls is not None:
        y = 50
        for name in ["thumb", "index", "middle", "ring", "little"]:
            val = curls.get(name, None)
            text = f"{name}: ----" if val is None else f"{name}: {val:.3f}"
            cv2.putText(frame, text, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 22


# ============================================================
# Main loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--show-3d", action="store_true")
    parser.add_argument("--depth-k", type=int, default=5)
    parser.add_argument("--max-hands", type=int, default=1)
    args = parser.parse_args()

    if args.show_3d and not OPEN3D_OK:
        print("[WARN] Open3D not available.")
        args.show_3d = False

    pipeline, align, intr = get_realsense_pipeline(args.width, args.height, args.fps)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    if args.show_3d:
        vis, pcd, line_set = opend3d_init()

    fps_hist = deque(maxlen=30)

    try:
        while True:
            t0 = time.time()
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            color_img = np.asanyarray(color.get_data())
            depth_raw = np.asanyarray(depth.get_data())
            depth_m = depth_raw * intr["depth_scale"]

            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            overlay = color_img.copy()

            if result.multi_hand_landmarks:
                for hand_lm, handedness in zip(result.multi_hand_landmarks,
                                               result.multi_handedness):

                    handed_raw = handedness.classification[0].label
                    handed = "Right" if handed_raw == "Left" else "Left"

                    lm_px = []
                    for lm in hand_lm.landmark:
                        u = int(np.clip(lm.x * intr["width"], 0, intr["width"] - 1))
                        v = int(np.clip(lm.y * intr["height"], 0, intr["height"] - 1))
                        lm_px.append((u, v))

                    xyz = np.full((21, 3), np.nan, dtype=np.float32)
                    for i, (u, v) in enumerate(lm_px):
                        z = median_depth_at(depth_m, u, v, k=args.depth_k)
                        if z is None or z <= 0:
                            continue
                        xyz[i] = backproject(u, v, z, intr)

                    curls = estimate_all_finger_curls(xyz)

                    fps_val = (1.0 / np.mean(fps_hist)) if len(fps_hist) > 5 else None
                    draw_2d_overlay(overlay, lm_px, handed, curls, fps_val)

                    if args.show_3d:
                        opend3d_update(vis, pcd, line_set, xyz)

                    break
            else:
                fps_val = (1.0 / np.mean(fps_hist)) if len(fps_hist) > 5 else None
                draw_2d_overlay(overlay, [None]*21, "None", curls=None, fps=fps_val)

            cv2.imshow("Hand Visualization", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            dt = time.time() - t0
            if dt > 0:
                fps_hist.append(dt)

    finally:
        pipeline.stop()
        hands.close()
        if args.show_3d:
            vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
