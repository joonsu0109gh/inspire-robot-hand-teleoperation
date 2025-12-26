#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    "thumb_base":  [1, 2],
    "thumb_tip":   [3, 4],
    "index":      [5, 6, 7, 8],
    "middle":     [9, 10, 11, 12],
    "ring":       [13, 14, 15, 16],
    "little":    [17, 18, 19, 20],
}
# Wrist index
WRIST_IDX = 0

INSPIRE_FINGER_IDX = {
    "little": 0,
    "ring": 1,
    "middle": 2,
    "index": 3,
    "thumb_tip": 4,
    "thumb_base": 5,
}


def compute_mcp_angle(landmarks_3d, mcp_idx, pip_idx, wrist_idx=0):
    """
    Compute MCP joint angle (in radians) using 3D landmarks.
    
    Returns angle in [0, π], where:
      0   = finger aligned with hand (fully extended)
      π/2 = ~90 degrees flexion
    """
    mcp = landmarks_3d[mcp_idx]
    pip = landmarks_3d[pip_idx]
    wrist = landmarks_3d[wrist_idx]

    if not (np.all(np.isfinite(mcp)) and np.all(np.isfinite(pip)) and np.all(np.isfinite(wrist))):
        return None

    # Vectors
    v1 = pip - mcp      # MCP → PIP
    v2 = mcp - wrist    # Wrist → MCP (palm direction)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    # Normalize
    v1 /= n1
    v2 /= n2

    # Angle
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot)   # radians

    return float(angle)


def estimate_mcp_angles(landmarks_3d):
    return {
        "index":  compute_mcp_angle(landmarks_3d, 5, 6),
        "middle": compute_mcp_angle(landmarks_3d, 9, 10),
        "ring":   compute_mcp_angle(landmarks_3d, 13, 14),
        "little": compute_mcp_angle(landmarks_3d, 17, 18),
        "thumb_tip":   compute_mcp_angle(landmarks_3d, 3, 4),
        "thumb_base":  compute_mcp_angle(landmarks_3d, 1, 2),
    }

# angle_min_max = {
#     "little": (0.28, 1.68),
#     "ring": (0.23, 1.45),
#     "middle": (0.05, 1.18),
#     "index": (0.17, 0.81),
#     "thumb_tip": (0.09, 1.5),
#     "thumb_base": (0.23, 0.65),
# }

# angle_min_max = {
#     "little": (0.38, 1.58),
#     "ring": (0.33, 1.35),
#     "middle": (0.15, 1.08),
#     "index": (0.27, 0.71),
#     "thumb_tip": (0.19, 1.4),
#     "thumb_base": (0.33, 0.55),
# }

angle_min_max = {
    "little": (0.4, 1.4),
    "ring": (0.4, 1.2),
    "middle": (0.2, 1.08),
    "index": (0.35, 0.71),
    "thumb_tip": (0.29, 1.4),
    "thumb_base": (0.23, 0.65),
}

def mcp_angles_to_joint_angles(mcp_angles):
    joint_angles = np.full(6, -1, dtype=int)


    for finger, idx in INSPIRE_FINGER_IDX.items():
        ang = mcp_angles[finger]
        if ang is None:
            joint_angles[idx] = -1
            continue

        # scale angle_min-angle_max to 1000-0
        ang_min, ang_max = angle_min_max[finger]
        ang_clamped = np.clip(ang, ang_min, ang_max)
        ang_scaled = (ang_clamped - ang_min) / (ang_max - ang_min)  # 0 to 1
        ang_final = (1.0 - ang_scaled) * 1000.0  # 1000 to 0

        joint_angles[idx] = int(ang_final)

    return joint_angles



def draw_overlay(frame_bgr, lm_px, handed_label, angles=None, fps=None):
    """
    Draw hand skeleton and some textual info on the image.

    lm_px: list of (u, v) pixel coords or None.
    angles: dict of finger joint angles (optional).
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
        
        cv2.putText(frame_bgr, str(idx), (pt[0] + 3, pt[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Handedness + FPS
    info = f"Hand: {handed_label}"
    if fps is not None:
        info += f" | FPS: {fps:.1f}"
    cv2.putText(frame_bgr, info, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2, cv2.LINE_AA)

    # Finger angle text (if available)
    if angles is not None:
        y0 = 45
        for name in INSPIRE_FINGER_IDX.keys():
            val = angles.get(name, None)
            txt = f"{name}: ----" if val is None else f"{name}: {val:.2f}"
            cv2.putText(frame_bgr, txt, (8, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 18


def median_filter_angles(mcp_angles, angle_hist):
    filtered = {}
    for finger, ang in mcp_angles.items():
        if ang is not None:
            angle_hist[finger].append(ang)
        if len(angle_hist[finger]) > 0:
            filtered[finger] = float(np.median(angle_hist[finger]))
        else:
            filtered[finger] = None
    return filtered

def threshold_filter(new_angles, prev_angles, threshold=0.05):
    """
    Only update if difference is larger than threshold (in radians).
    new_angles, prev_angles = dict {finger: angle (rad)}
    """
    filtered = {}

    for finger, ang_new in new_angles.items():
        ang_prev = prev_angles.get(finger, None)

        # 이전 기록이 없다면 바로 채택
        if ang_prev is None:
            filtered[finger] = ang_new
            continue

        # None 값 처리
        if ang_new is None:
            filtered[finger] = ang_prev
            continue

        diff = abs(ang_new - ang_prev)

        # threshold 이하면 업데이트하지 않음 → prev 유지
        if diff < threshold:
            filtered[finger] = ang_prev
        else:
            filtered[finger] = ang_new

    return filtered


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
    last_send_time = 0.0

    prev_mcp_angles = {finger: None for finger in INSPIRE_FINGER_IDX.keys()}
    angle_hist = {finger: deque(maxlen=10) for finger in INSPIRE_FINGER_IDX.keys()}


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
                    mcp_angles = estimate_mcp_angles(xyz)
                    # mcp_angles = smooth_mcp_angles(mcp_angles, prev_mcp_angles, alpha=0.1)
                    mcp_angles = median_filter_angles(mcp_angles, angle_hist)

                    mcp_angles = threshold_filter(mcp_angles, prev_mcp_angles)

                    # for smoothing, could add temporal filtering here
                    
                    prev_mcp_angles = mcp_angles 

                    fps_val = (1.0 / np.mean(fps_hist)) if len(fps_hist) > 5 else None
                    draw_overlay(overlay, lm_px, handed_label, angles=mcp_angles, fps=fps_val)

                    now = time.time()
                    if mcp_angles is not None and (now - last_send_time) >= args.send_interval:
                        
                        joint_angles = mcp_angles_to_joint_angles(mcp_angles)
                        print(f"[Teleop] angleSet: {joint_angles}")

                        write6(client, "angleSet", joint_angles.tolist())
                        last_send_time = now
                    # For now, handle only the first detected hand
                    break
            else:
                fps_val = (1.0 / np.mean(fps_hist)) if len(fps_hist) > 5 else None
                draw_overlay(overlay, [None] * 21, handed_label, angles=None, fps=fps_val)

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
