# Tactile Robot Hand Teleoperation

**Inspire-Robots RH56DFTP – Hand Pose Estimation-based Teleoperation**

This repository presents a tactile robot hand teleoperation system based on **hand pose estimation** and **tactile sensing visualization**. The system enables intuitive human-to-robot control using finger angles with 6-DoF mapping.

---

## Environment

* **OS**: Ubuntu 22.04
* **Camera**: Intel RealSense D435
* **Robot Hand**: Inspire-Robots RH56DFTP

---

## Demo Video

[![Tactile Robot Hand Teleoperation Demo](https://img.youtube.com/vi/8WQP30CKw1Q/hqdefault.jpg)](https://youtu.be/8WQP30CKw1Q)

> Click the thumbnail above to watch the full demonstration on YouTube.

---

## Usage

### 1. Visualize Tactile Sensing

Visualize tactile sensor data from the robot hand in real time.

```shell
python touch_data_visualization.py
```

---

### 2. Hand Pose Estimation-based Robot Hand Teleoperation

Teleoperate the robot hand using hand pose estimation.

* **Control Method**: Finger angle-based 6-DoF control

```shell
python teleop_robot_hand_angle.py
```
