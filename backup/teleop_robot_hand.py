#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time Inspire hand teleoperation (retargeting) using:
- RealSense + MediaPipe for human hand tracking
- Simple finger curl-based retargeting
- Modbus TCP control of Inspire hand

Pipeline:
  RealSense RGB-D + MediaPipe
    -> 3D landmarks (camera frame)
    -> per-finger curl estimation (0.0 = fully open, 1.0 = fully closed)
    -> map curl to Inspire hand angleSet values (0 ~ 1000)
    -> send angleSet via Modbus (demo_modbus.write6)

Notes:
- This is a simple baseline: one scalar curl per finger.
- Joint-level IK-based retargeting using `InspireHandKinematics` can be added later
  if you want more accurate fingertip pose mapping.
"""

import argparse
import time
from collections import deque

import numpy as np
import cv2
import mediapipe as mp

# Import existing utilities
from demo_modbus import open_modbus, write6  # :contentReference[oaicite:3]{index=3}
from hand_kinematics_system import InspireHandKinematics  # (optional for future use) :contentReference[oaicite:4]{index=4}
from realsense_hand_pose_viz import (  # :contentReference[oaicite:5]{index=5}
    get_realsense_pipeline,
    median_depth_at,
    backproject,
)


mp_hands = mp.solutions.hands

# MediaPipe hand landmark indices per finger
FINGER_LANDMARKS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "little": [17, 18, 19, 20],
}
# Wrist index
WRIST_IDX = 0

# Mapping from finger name to Inspire angleSet index
# TODO: Adjust this mapping to match your actual Inspire hand configuration.
ANGLESET_INDEX = {
    "little": 0,
    "ring": 1,
    "middle": 2,
    "index": 3,
    "thumb_base": 4,
    "thumb_tip": 5,
}


def compute_finger_curl(landmarks_3d, finger_name):
    """
    Compute a scalar curl value in [0, 1] for a single finger.

    Curl metric:
      - Use vectors from MCP(=first) to PIP(=second), and MCP to TIP.
      - curl = 1 - (dot(v1, v2) + 1) / 2
        -> extended finger: dot ~ 1 -> curl ~ 0
        -> fully bent finger: dot ~ -1 -> curl ~ 1

    landmarks_3d: (21, 3) array in meters, may contain NaNs.
    finger_name: one of ['thumb', 'index', 'middle', 'ring', 'little']
    """
    idxs = FINGER_LANDMARKS[finger_name]
    mcp_idx, pip_idx, dip_idx, tip_idx = idxs  # we mainly use mcp, pip, tip

    mcp = landmarks_3d[mcp_idx]
    pip = landmarks_3d[pip_idx]
    tip = landmarks_3d[tip_idx]

    if not (np.all(np.isfinite(mcp)) and np.all(np.isfinite(pip)) and np.all(np.isfinite(tip))):
        return None

    v1 = pip - mcp
    v2 = tip - mcp

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return None

    v1 /= norm1
    v2 /= norm2

    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    curl = 1.0 - (dot + 1.0) / 2.0  # map [-1, 1] -> [1, 0] then invert

    curl = float(np.clip(curl, 0.0, 1.0))
    return curl


def estimate_all_finger_curls(landmarks_3d):
    """
    Estimate curl values for all fingers.

    Returns:
        dict: {finger_name: curl or None}
    """
    curls = {}
    for name in FINGER_LANDMARKS.keys():
        curls[name] = compute_finger_curl(landmarks_3d, name)
    return curls


def curls_to_angleset(curls, prev_angleset, smooth_alpha=0.4):
    """
    Map curl values to Inspire hand angleSet.
    New mapping order:
        0: little
        1: ring
        2: middle
        3: index
        4: thumb base
        5: thumb tip
    """
    angleset = np.full(6, -1, dtype=int)

    # Simple fingers (1 DOF each)
    finger_mapping = {
        "little": 0,
        "ring": 1,
        "middle": 2,
        "index": 3,
    }

    for finger_name, idx in finger_mapping.items():
        curl = curls.get(finger_name, None)
        if curl is None:
            angleset[idx] = int(prev_angleset[idx])
            continue

        target = int(np.clip((1.0 - curl) * 1000.0, 0, 1000))

        old = prev_angleset[idx]
        if old < 0:
            smoothed = target
        else:
            smoothed = (1 - smooth_alpha) * target + smooth_alpha * old

        angleset[idx] = int(smoothed)

    # Thumb (2 DOF) — both use curl for now
    thumb_curl = curls.get("thumb", None)
    if thumb_curl is None:
        angleset[4] = int(prev_angleset[4])
        angleset[5] = int(prev_angleset[5])
    else:
        thumb_target = int(np.clip((1.0 - thumb_curl) * 1000.0, 0, 1000))

        # Thumb base
        old = prev_angleset[4]
        angleset[4] = thumb_target if old < 0 else int((1 - smooth_alpha) * thumb_target + smooth_alpha * old)

        # Thumb tip — you can scale differently if you want:
        # e.g., tip = 0.7 * base
        tip_target = thumb_target   # currently same mapping
        old2 = prev_angleset[5]
        angleset[5] = tip_target if old2 < 0 else int((1 - smooth_alpha) * tip_target + smooth_alpha * old2)

    return angleset



def draw_overlay(frame_bgr, lm_px, handed_label, curls=None, fps=None):
    """
    Draw hand skeleton and some textual info on the image.

    lm_px: list of (u, v) pixel coords or None.
    curls: dict of finger curl values (optional).
    """
    h, w = frame_bgr.shape[:2]

    # Draw skeleton
    for a, b in mp_hands.HAND_CONNECTIONS:
        if lm_px[a] is None or lm_px[b] is None:
            continue
        cv2.line(frame_bgr, lm_px[a], lm_px[b], (0, 215, 255), 2)

    for idx, pt in enumerate(lm_px):
        if pt is None:
            continue
        cv2.circle(frame_bgr, pt, 3, (255, 170, 30), -1)
        if idx in (0, 5, 9, 13, 17):
            cv2.putText(frame_bgr, str(idx), (pt[0] + 3, pt[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Handedness + FPS
    info = f"Hand: {handed_label}"
    if fps is not None:
        info += f" | FPS: {fps:.1f}"
    cv2.putText(frame_bgr, info, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2, cv2.LINE_AA)

    # Finger curls text (if available)
    if curls is not None:
        y0 = 45
        for name in ["thumb", "index", "middle", "ring", "little"]:
            val = curls.get(name, None)
            txt = f"{name}: ----" if val is None else f"{name}: {val:.2f}"
            cv2.putText(frame_bgr, txt, (8, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 18


def main():
    parser = argparse.ArgumentParser(
        description="Inspire hand teleoperation via RealSense + MediaPipe (curl retargeting)"
    )
    parser.add_argument("--ip", type=str, default="192.168.11.210", help="Modbus TCP IP of Inspire hand")
    parser.add_argument("--port", type=int, default=6000, help="Modbus TCP port of Inspire hand")
    parser.add_argument("--width", type=int, default=1280, help="RealSense color/depth width")
    parser.add_argument("--height", type=int, default=720, help="RealSense color/depth height")
    parser.add_argument("--fps", type=int, default=30, help="RealSense FPS")
    parser.add_argument("--max-hands", type=int, default=1, help="Max number of hands to track")
    parser.add_argument("--depth-k", type=int, default=5, help="Median window size for depth sampling")
    parser.add_argument("--send-interval", type=float, default=0.05,
                        help="Minimum time (sec) between Modbus angleSet commands")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1) Connect to Inspire hand via Modbus
    # ------------------------------------------------------------------
    print("[Teleop] Opening Modbus TCP connection...")
    client = open_modbus(args.ip, args.port)
    print("[Teleop] Modbus connected.")

    # Set speed and force to reasonable defaults. Tune as needed.
    print("[Teleop] Setting speedSet and forceSet.")
    write6(client, "speedSet", [800, 800, 800, 800, 800, 800])
    time.sleep(0.5)
    write6(client, "forceSet", [500, 500, 500, 500, 500, 500])
    time.sleep(0.5)
    print("[Teleop] Initial parameters set.")

    # Optional: initialize kinematics system for future IK-based retargeting
    # IK not used in this simple curl-based demo
    try:
        kin_system = InspireHandKinematics()
        print("[Teleop] InspireHandKinematics loaded.")
    except Exception as exc:
        print(f"[Teleop] Warning: could not initialize InspireHandKinematics: {exc}")
        kin_system = None

    # ------------------------------------------------------------------
    # 2) Setup RealSense + MediaPipe
    # ------------------------------------------------------------------
    pipeline, align, intr = get_realsense_pipeline(args.width, args.height, args.fps)
    print("[Teleop] RealSense pipeline started.")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    fps_hist = deque(maxlen=30)
    prev_angleset = np.full(6, -1, dtype=float)
    last_send_time = 0.0

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
            depth_raw = np.asanyarray(depth.get_data())  # uint16
            depth_m = depth_raw * intr["depth_scale"]    # float meters

            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            overlay = color_img.copy()
            curls = None
            handed_label = "None"

            if result.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(
                    result.multi_hand_landmarks, result.multi_handedness
                ):
                    # MediaPipe handedness is person-centric; flip for camera view
                    handed_raw = hand_handedness.classification[0].label
                    handed_label = "Right" if handed_raw == "Left" else "Left"

                    # Collect 2D pixel coordinates
                    lm_px = []
                    for lm in hand_landmarks.landmark:
                        u = int(np.clip(lm.x * intr["width"], 0, intr["width"] - 1))
                        v = int(np.clip(lm.y * intr["height"], 0, intr["height"] - 1))
                        lm_px.append((u, v))

                    # Back-project to 3D
                    xyz = np.full((21, 3), np.nan, dtype=np.float32)
                    for i, (u, v) in enumerate(lm_px):
                        z = median_depth_at(depth_m, u, v, k=args.depth_k)
                        if z is None or z <= 0:
                            continue
                        xyz[i] = backproject(u, v, z, intr)

                    # Compute finger curls
                    curls = estimate_all_finger_curls(xyz)

                    # Draw overlay
                    fps_val = (1.0 / np.mean(fps_hist)) if len(fps_hist) > 5 else None
                    draw_overlay(overlay, lm_px, handed_label, curls=curls, fps=fps_val)

                    # Convert curls to angleSet and send via Modbus (rate-limited)
                    now = time.time()
                    if curls is not None and (now - last_send_time) >= args.send_interval:
                        angleset = curls_to_angleset(curls, prev_angleset)
                        # Debug print for inspection
                        print(f"[Teleop] angleSet: {angleset}")
                        write6(client, "angleSet", angleset.tolist())
                        prev_angleset = angleset.astype(float)
                        last_send_time = now

                    # For now, handle only the first detected hand
                    break
            else:
                fps_val = (1.0 / np.mean(fps_hist)) if len(fps_hist) > 5 else None
                draw_overlay(overlay, [None] * 21, handed_label, curls=None, fps=fps_val)

            cv2.imshow("Inspire Hand Teleop (2D View)", overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[Teleop] Quit requested by user.")
                break

            # FPS accounting
            dt = time.time() - t0
            if dt > 0:
                fps_hist.append(dt)

    except KeyboardInterrupt:
        print("[Teleop] Interrupted by user.")
    finally:
        print("[Teleop] Shutting down...")
        hands.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        client.close()
        print("[Teleop] Closed RealSense, MediaPipe, and Modbus connection.")


if __name__ == "__main__":
    main()
