#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from collections import deque

import numpy as np
import cv2
import mediapipe as mp

# External utilities
from demo_modbus import open_modbus, write6
from hand_kinematics_system import InspireHandKinematics  # optional
from realsense_hand_pose_viz import (
    get_realsense_pipeline,
    median_depth_at,
    backproject,
)

mp_hands = mp.solutions.hands

# ============================================================
# MediaPipe landmark indices
# ============================================================

FINGER_LANDMARKS = {
    "thumb_base":  [1, 2],
    "thumb_tip":   [3, 4],
    "index":       [5, 6, 7, 8],
    "middle":      [9, 10, 11, 12],
    "ring":        [13, 14, 15, 16],
    "little":      [17, 18, 19, 20],
}

WRIST_IDX = 0

# Inspire hand joint index mapping
INSPIRE_FINGER_IDX = {
    "little": 0,
    "ring": 1,
    "middle": 2,
    "index": 3,
    "thumb_tip": 4,
    "thumb_base": 5,
}

# ============================================================
# MCP angle estimation
# ============================================================

def compute_mcp_angle(landmarks_3d, mcp_idx, pip_idx, wrist_idx=0):
    """
    Compute MCP joint flexion angle (radians) using 3D landmarks.

    Angle definition:
        0      : fully extended
        pi / 2 : ~90 degree flexion
    """
    mcp = landmarks_3d[mcp_idx]
    pip = landmarks_3d[pip_idx]
    wrist = landmarks_3d[wrist_idx]

    if not np.all(np.isfinite([*mcp, *pip, *wrist])):
        return None

    v1 = pip - mcp      # MCP -> PIP
    v2 = mcp - wrist    # Wrist -> MCP (palm direction)

    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    v1 /= n1
    v2 /= n2

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.arccos(dot))


def estimate_mcp_angles(landmarks_3d):
    """Estimate MCP angles for all fingers."""
    return {
        "index":       compute_mcp_angle(landmarks_3d, 5, 6),
        "middle":      compute_mcp_angle(landmarks_3d, 9, 10),
        "ring":        compute_mcp_angle(landmarks_3d, 13, 14),
        "little":      compute_mcp_angle(landmarks_3d, 17, 18),
        "thumb_tip":   compute_mcp_angle(landmarks_3d, 3, 4),
        "thumb_base":  compute_mcp_angle(landmarks_3d, 1, 2),
    }


# ============================================================
# Angle normalization (experimentally tuned)
# ============================================================

angle_min_max = {
    "little": (0.4, 1.4),
    "ring": (0.4, 1.2),
    "middle": (0.2, 1.08),
    "index": (0.35, 0.71),
    "thumb_tip": (0.29, 1.4),
    "thumb_base": (0.23, 0.65),
}


def mcp_angles_to_joint_angles(mcp_angles):
    """
    Convert MCP angles (radians) to Inspire hand joint command space.

    Output:
        np.ndarray of shape (6,)
        Range: [0, 1000], -1 if invalid
    """
    joint_angles = np.full(6, -1, dtype=int)

    for finger, idx in INSPIRE_FINGER_IDX.items():
        ang = mcp_angles.get(finger, None)
        if ang is None:
            continue

        ang_min, ang_max = angle_min_max[finger]
        ang_clamped = np.clip(ang, ang_min, ang_max)
        norm = (ang_clamped - ang_min) / (ang_max - ang_min)
        joint_angles[idx] = int((1.0 - norm) * 1000.0)

    return joint_angles


# ============================================================
# Visualization utilities
# ============================================================

def draw_overlay(frame_bgr, lm_px, handed_label, angles=None, fps=None):
    """
    Draw hand skeleton, landmark indices, and diagnostic text.
    """
    for a, b in mp_hands.HAND_CONNECTIONS:
        if lm_px[a] and lm_px[b]:
            cv2.line(frame_bgr, lm_px[a], lm_px[b], (0, 215, 255), 2)

    for idx, pt in enumerate(lm_px):
        if pt is None:
            continue
        cv2.circle(frame_bgr, pt, 3, (255, 170, 30), -1)
        cv2.putText(frame_bgr, str(idx), (pt[0] + 3, pt[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    header = f"Hand: {handed_label}"
    if fps:
        header += f" | FPS: {fps:.1f}"
    cv2.putText(frame_bgr, header, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2)

    if angles:
        y = 45
        for k in INSPIRE_FINGER_IDX:
            v = angles.get(k, None)
            txt = f"{k}: ----" if v is None else f"{k}: {v:.2f}"
            cv2.putText(frame_bgr, txt, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 18


# ============================================================
# Temporal filtering
# ============================================================

def median_filter_angles(mcp_angles, angle_hist):
    """Apply median filter over short temporal window."""
    out = {}
    for finger, ang in mcp_angles.items():
        if ang is not None:
            angle_hist[finger].append(ang)
        out[finger] = float(np.median(angle_hist[finger])) if angle_hist[finger] else None
    return out


def threshold_filter(new_angles, prev_angles, threshold=0.05):
    """
    Update angles only if change exceeds threshold (radians).
    """
    out = {}
    for finger, ang_new in new_angles.items():
        ang_prev = prev_angles.get(finger, None)

        if ang_prev is None or ang_new is None:
            out[finger] = ang_new if ang_new is not None else ang_prev
            continue

        out[finger] = ang_new if abs(ang_new - ang_prev) >= threshold else ang_prev

    return out


# ============================================================
# Main application
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inspire Hand Teleoperation via RealSense + MediaPipe"
    )
    parser.add_argument("--ip", type=str, default="192.168.11.210")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--depth-k", type=int, default=5)
    parser.add_argument("--send-interval", type=float, default=0.05)
    args = parser.parse_args()

    print("[Teleop] Connecting to Inspire hand...")
    client = open_modbus(args.ip, args.port)

    write6(client, "speedSet", [800] * 6)
    time.sleep(0.5)
    write6(client, "forceSet", [500] * 6)
    time.sleep(0.5)

    try:
        kin_system = InspireHandKinematics()
        print("[Teleop] Kinematics system loaded.")
    except Exception as e:
        print(f"[Teleop] Kinematics unavailable: {e}")
        kin_system = None

    pipeline, align, intr = get_realsense_pipeline(
        args.width, args.height, args.fps
    )
    print("[Teleop] RealSense pipeline started.")

    hands = mp_hands.Hands(
        max_num_hands=args.max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    fps_hist = deque(maxlen=30)
    angle_hist = {k: deque(maxlen=10) for k in INSPIRE_FINGER_IDX}
    prev_angles = {k: None for k in INSPIRE_FINGER_IDX}
    last_send = 0.0

    try:
        while True:
            t0 = time.time()
            frames = align.process(pipeline.wait_for_frames())
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue

            color_img = np.asanyarray(color.get_data())
            depth_m = np.asanyarray(depth.get_data()) * intr["depth_scale"]

            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            overlay = color_img.copy()
            handed = "None"

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                handed_raw = result.multi_handedness[0].classification[0].label
                handed = "Right" if handed_raw == "Left" else "Left"

                lm_px = []
                xyz = np.full((21, 3), np.nan)

                for i, p in enumerate(lm.landmark):
                    u = int(np.clip(p.x * intr["width"], 0, intr["width"] - 1))
                    v = int(np.clip(p.y * intr["height"], 0, intr["height"] - 1))
                    lm_px.append((u, v))

                    z = median_depth_at(depth_m, u, v, args.depth_k)
                    if z:
                        xyz[i] = backproject(u, v, z, intr)

                mcp = estimate_mcp_angles(xyz)
                mcp = median_filter_angles(mcp, angle_hist)
                mcp = threshold_filter(mcp, prev_angles)
                prev_angles = mcp

                now = time.time()
                if now - last_send >= args.send_interval:
                    joints = mcp_angles_to_joint_angles(mcp)
                    write6(client, "angleSet", joints.tolist())
                    last_send = now

                fps = 1.0 / np.mean(fps_hist) if len(fps_hist) > 5 else None
                draw_overlay(overlay, lm_px, handed, mcp, fps)

            cv2.imshow("Inspire Hand Teleop", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            fps_hist.append(time.time() - t0)

    finally:
        print("[Teleop] Shutting down...")
        hands.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        client.close()


if __name__ == "__main__":
    main()
