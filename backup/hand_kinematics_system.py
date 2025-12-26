#!/usr/bin/env python3
"""
Utility to visualize and control the Inspire dexterous hand using forward
and inverse kinematics derived from the URDF description.

The script exposes a small CLI:
  - FK mode: compute poses for any set of joints and optionally plot the hand.
  - IK mode: solve for joint angles that realize a desired fingertip pose.

Examples:
python3 hand_kinematics_system.py fk --finger index \
    --joint right_index_1_joint=0.2 --joint right_index_2_joint=0.1 --plot

python3 hand_kinematics_system.py ik --finger thumb \
    --target 0.05 0.02 0.15 --orientation-mode Z --orientation 0 0 1
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
from ikpy.chain import Chain


warnings.filterwarnings(
    "ignore",
    message="Link .* is of type 'fixed' but set as active.*",
    module="ikpy.chain",
)
warnings.filterwarnings(
    "ignore",
    message="Joint .* is of type: fixed, but has an 'axis' attribute defined.*",
    module="ikpy.urdf.URDF",
)


SDK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = (
    SDK_ROOT
    / "URDF"
    / "4th generation"
    / "urdf_with_force_sensor_ts"
    / "src"
    / "urdf"
    / "urdf_with_force_sensor_ts.urdf"
)

FINGER_TIP_LINKS = {
    "thumb": "thumb_force_sensor",
    "index": "index_force_sensor",
    "middle": "middle_force_sensor",
    "ring": "ring_force_sensor",
    "little": "little_force_sensor",
}


@dataclass
class FingerChain:
    """Convenience wrapper for each finger specific chain."""

    name: str
    tip_link: str
    base_sequence: List[str]
    chain: Chain
    joint_indices: List[int]
    joint_names: List[str]
    joint_limits: Dict[str, Tuple[float, float]]

    def vector_from_state(self, joint_state: Mapping[str, float]) -> np.ndarray:
        """Build the full joint vector expected by ikpy from a sparse dict."""
        joints = np.zeros(len(self.chain.links), dtype=float)
        for idx, name in zip(self.joint_indices, self.joint_names):
            joints[idx] = joint_state.get(name, 0.0)
        return joints

    def map_solution(self, solution_vector: Sequence[float]) -> Dict[str, float]:
        """Extract actuated joint values from a full IK solution vector."""
        result = {}
        for idx, name in zip(self.joint_indices, self.joint_names):
            value = float(solution_vector[idx])
            lower, upper = self.joint_limits[name]
            if not math.isinf(lower):
                value = max(lower, value)
            if not math.isinf(upper):
                value = min(upper, value)
            result[name] = value
        return result


class InspireHandKinematics:
    """Loads the Inspire hand URDF and exposes FK/IK helpers per finger."""

    def __init__(
        self,
        urdf_path: Path = DEFAULT_URDF,
        finger_tip_links: Mapping[str, str] = FINGER_TIP_LINKS,
    ) -> None:
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.urdf_path = urdf_path
        self._tree = ET.parse(urdf_path)
        self._root = self._tree.getroot()
        self.base_link = self._detect_base_link()
        self.fingers: Dict[str, FingerChain] = {}
        for finger_name, tip_link in finger_tip_links.items():
            base_sequence = self._build_base_sequence(tip_link)
            chain = Chain.from_urdf_file(
                str(self.urdf_path),
                base_elements=base_sequence,
                name=f"{finger_name}_chain",
            )
            joint_indices: List[int] = []
            joint_names: List[str] = []
            joint_limits: Dict[str, Tuple[float, float]] = {}
            for idx, link in enumerate(chain.links):
                if link.joint_type == "fixed":
                    continue
                joint_indices.append(idx)
                joint_names.append(link.name)
                joint_limits[link.name] = tuple(link.bounds)

            # Activate only movable joints for IK
            active_mask = np.zeros(len(chain.links), dtype=bool)
            for idx in joint_indices:
                active_mask[idx] = True
            chain.active_links_mask = active_mask

            self.fingers[finger_name] = FingerChain(
                name=finger_name,
                tip_link=tip_link,
                base_sequence=base_sequence,
                chain=chain,
                joint_indices=joint_indices,
                joint_names=joint_names,
                joint_limits=joint_limits,
            )

    def forward_kinematics(
        self,
        finger: str,
        joint_state: Mapping[str, float],
        full_kinematics: bool = False,
    ) -> np.ndarray | List[np.ndarray]:
        """Compute FK for the selected finger."""
        finger_chain = self._get_finger(finger)
        joints = finger_chain.vector_from_state(joint_state)
        return finger_chain.chain.forward_kinematics(
            joints=joints, full_kinematics=full_kinematics
        )

    def inverse_kinematics(
        self,
        finger: str,
        target_position: Sequence[float],
        target_orientation: Optional[np.ndarray] = None,
        orientation_mode: Optional[str] = None,
        initial_state: Optional[Mapping[str, float]] = None,
        regularization: Optional[float] = 1e-3,
        position_tolerance: float = 5e-4,
    ) -> Dict[str, float]:
        """Solve IK for a fingertip pose."""
        finger_chain = self._get_finger(finger)
        target_position = np.asarray(target_position, dtype=float)
        seeds: List[np.ndarray] = []
        if initial_state:
            seeds.append(finger_chain.vector_from_state(initial_state))
        seeds.append(self._default_seed_vector(finger_chain))
        seeds.append(finger_chain.vector_from_state({}))

        tried_errors: List[float] = []
        for seed in _unique_seed_vectors(seeds):
            try:
                solution, error = self._run_ikpy_attempt(
                    finger_chain=finger_chain,
                    target_position=target_position,
                    target_orientation=target_orientation,
                    orientation_mode=orientation_mode,
                    seed=seed,
                    regularization=regularization,
                )
            except Exception:  # pylint: disable=broad-except
                continue
            if error <= position_tolerance:
                return finger_chain.map_solution(solution)
            tried_errors.append(float(error))

        if orientation_mode is not None:
            raise RuntimeError(
                "IKPy could not satisfy the requested pose and "
                "the fallback solver does not enforce orientation constraints. "
                f"Residual errors from IKPy attempts: {tried_errors}"
            )

        damping = regularization if regularization not in (None, 0.0) else 1e-3
        for seed in _unique_seed_vectors(seeds):
            solution = self._solve_dls(
                finger_chain=finger_chain,
                target_position=target_position,
                seed=seed,
                damping=damping,
                tolerance=position_tolerance,
            )
            if solution is not None:
                return finger_chain.map_solution(solution)

        raise RuntimeError(
            f"Unable to reach target within {position_tolerance} m. "
            f"Residual errors from attempts: {tried_errors}"
        )

    def visualize(
        self,
        joint_state: Mapping[str, float],
        highlight_finger: Optional[str] = None,
        target_points: Optional[Iterable[Sequence[float]]] = None,
    ) -> None:
        """Render the hand configuration using Matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for visualization") from exc

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = {
            "thumb": "tab:red",
            "index": "tab:blue",
            "middle": "tab:green",
            "ring": "tab:orange",
            "little": "tab:purple",
        }
        for name, finger in self.fingers.items():
            joints = finger.vector_from_state(joint_state)
            frames = finger.chain.forward_kinematics(
                joints=joints, full_kinematics=True
            )
            xyz = np.array([[0.0, 0.0, 0.0]])
            xyz = np.vstack([xyz, [[frame[0, 3], frame[1, 3], frame[2, 3]] for frame in frames]])
            label = f"{name} finger"
            color = colors.get(name, None)
            linewidth = 3.0 if name == highlight_finger else 1.5
            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                marker="o",
                label=label,
                color=color,
                linewidth=linewidth,
            )
        if target_points:
            targets = np.array(list(target_points))
            ax.scatter(
                targets[:, 0],
                targets[:, 1],
                targets[:, 2],
                marker="x",
                color="black",
                s=80,
                label="targets",
            )
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.legend()
        self._set_equal_axes(ax)
        plt.tight_layout()
        plt.show()

    def pretty_pose(self, transform: np.ndarray) -> str:
        """Return a readable string for a 4x4 transform."""
        position = transform[:3, 3]
        rpy = self._rotation_to_rpy(transform[:3, :3])
        return (
            f"pos = ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m, "
            f"rpy = ({rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}) rad"
        )

    def joint_order(self, finger: str) -> List[str]:
        """Expose the joint ordering for CLI users."""
        return list(self._get_finger(finger).joint_names)

    def _default_seed_vector(self, finger_chain: FingerChain) -> np.ndarray:
        """Generate a neutral IK seed at the midpoint of each joint limit."""
        seed_state: Dict[str, float] = {}
        for name in finger_chain.joint_names:
            lower, upper = finger_chain.joint_limits[name]
            if math.isinf(lower) or math.isinf(upper):
                seed_state[name] = 0.0
            else:
                seed_state[name] = 0.5 * (lower + upper)
        return finger_chain.vector_from_state(seed_state)

    def _get_finger(self, finger: str) -> FingerChain:
        if finger not in self.fingers:
            raise KeyError(f"Unknown finger '{finger}'. Expected one of {list(self.fingers)}")
        return self.fingers[finger]

    def _detect_base_link(self) -> str:
        all_links = {link.attrib["name"] for link in self._root.findall("link")}
        child_links = {joint.find("child").attrib["link"] for joint in self._root.findall("joint")}
        base_candidates = all_links - child_links
        if not base_candidates:
            raise ValueError("Unable to detect base link from URDF")
        return sorted(base_candidates)[0]

    def _build_base_sequence(self, tip_link: str) -> List[str]:
        child_to_joint: Dict[str, Tuple[str, str]] = {}
        for joint in self._root.findall("joint"):
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            child_to_joint[child] = (joint.attrib["name"], parent)

        sequence: List[str] = [self.base_link]
        chain: List[Tuple[str, str]] = []
        cursor = tip_link
        while cursor in child_to_joint:
            joint_name, parent = child_to_joint[cursor]
            chain.append((joint_name, cursor))
            cursor = parent

        if cursor != self.base_link:
            raise ValueError(f"Tip '{tip_link}' is not connected to base link '{self.base_link}'")

        for joint_name, link_name in reversed(chain):
            sequence.extend([joint_name, link_name])
        return sequence

    @staticmethod
    def _rotation_to_rpy(rot: np.ndarray) -> Tuple[float, float, float]:
        """Convert a rotation matrix to roll-pitch-yaw (XYZ intrinsic)."""
        sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(rot[2, 1], rot[2, 2])
            pitch = math.atan2(-rot[2, 0], sy)
            yaw = math.atan2(rot[1, 0], rot[0, 0])
        else:
            roll = math.atan2(-rot[1, 2], rot[1, 1])
            pitch = math.atan2(-rot[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    @staticmethod
    def _set_equal_axes(ax) -> None:
        """Apply equal scaling for 3D plots."""
        ranges = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        span = ranges[:, 1] - ranges[:, 0]
        center = np.mean(ranges, axis=1)
        max_span = max(span)
        limits = np.array([
            center - max_span / 2,
            center + max_span / 2,
        ])
        ax.set_xlim3d(limits[0, 0], limits[1, 0])
        ax.set_ylim3d(limits[0, 1], limits[1, 1])
        ax.set_zlim3d(limits[0, 2], limits[1, 2])

    def _run_ikpy_attempt(
        self,
        finger_chain: FingerChain,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        orientation_mode: Optional[str],
        seed: np.ndarray,
        regularization: Optional[float],
    ) -> Tuple[np.ndarray, float]:
        solution = finger_chain.chain.inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation,
            orientation_mode=orientation_mode,
            initial_position=seed,
            regularization_parameter=regularization,
        )
        fk = finger_chain.chain.forward_kinematics(solution)
        error = np.linalg.norm(fk[:3, 3] - target_position)
        return solution, float(error)

    def _solve_dls(
        self,
        finger_chain: FingerChain,
        target_position: np.ndarray,
        seed: np.ndarray,
        damping: float,
        tolerance: float,
        max_iters: int = 200,
        step_limit: float = 0.35,
    ) -> Optional[np.ndarray]:
        """Damped least squares IK fallback (position only)."""
        q = seed.copy()
        for _ in range(max_iters):
            fk = finger_chain.chain.forward_kinematics(q)
            error_vec = target_position - fk[:3, 3]
            if np.linalg.norm(error_vec) <= tolerance:
                return q
            jac = self._finite_difference_jacobian(finger_chain, q)
            lhs = jac @ jac.T + (damping ** 2) * np.eye(3)
            try:
                delta_task = jac.T @ np.linalg.solve(lhs, error_vec)
            except np.linalg.LinAlgError:
                return None
            delta_task = np.clip(delta_task, -step_limit, step_limit)
            for local_idx, joint_idx in enumerate(finger_chain.joint_indices):
                q[joint_idx] += float(delta_task[local_idx])
            self._apply_joint_limits(finger_chain, q)
        return None

    def _finite_difference_jacobian(
        self,
        finger_chain: FingerChain,
        joints_vector: np.ndarray,
        delta: float = 1e-4,
    ) -> np.ndarray:
        """Approximate the translational Jacobian numerically."""
        base_fk = finger_chain.chain.forward_kinematics(joints_vector)
        base_pos = base_fk[:3, 3]
        jac = np.zeros((3, len(finger_chain.joint_indices)))
        for col, joint_idx in enumerate(finger_chain.joint_indices):
            perturbed = joints_vector.copy()
            perturbed[joint_idx] += delta
            fk = finger_chain.chain.forward_kinematics(perturbed)
            jac[:, col] = (fk[:3, 3] - base_pos) / delta
        return jac

    def _apply_joint_limits(
        self,
        finger_chain: FingerChain,
        joints_vector: np.ndarray,
    ) -> None:
        """Clamp joint values to their URDF limits."""
        for idx, name in zip(finger_chain.joint_indices, finger_chain.joint_names):
            lower, upper = finger_chain.joint_limits[name]
            value = joints_vector[idx]
            if not math.isinf(lower):
                value = max(lower, value)
            if not math.isinf(upper):
                value = min(upper, value)
            joints_vector[idx] = value


def parse_joint_overrides(overrides: Optional[Iterable[str]]) -> Dict[str, float]:
    """Parse --joint key=value CLI options."""
    result: Dict[str, float] = {}
    if not overrides:
        return result
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid joint override '{item}', expected name=value")
        name, raw_value = item.split("=", 1)
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid joint value '{raw_value}' for joint '{name}'") from exc
        result[name.strip()] = value
    return result


def _unique_seed_vectors(seeds: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Remove duplicate seed guesses."""
    unique: List[np.ndarray] = []
    for seed in seeds:
        if seed is None:
            continue
        if not unique:
            unique.append(seed)
            continue
        if any(np.allclose(seed, existing) for existing in unique):
            continue
        unique.append(seed)
    return unique


def parse_orientation(values: Optional[Sequence[float]], mode: Optional[str]) -> Optional[np.ndarray]:
    """Interpret orientation arguments given the selected mode."""
    if mode is None:
        return None
    if values is None:
        raise ValueError("An orientation vector must be provided when orientation-mode is set")
    arr = np.asarray(values, dtype=float)
    if mode in {"X", "Y", "Z"}:
        if arr.shape[0] != 3:
            raise ValueError("Orientation vectors must contain exactly 3 numbers")
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Orientation vector must be non-zero")
        return arr / norm
    if mode == "all":
        if arr.shape[0] != 9:
            raise ValueError("Orientation matrix needs 9 values (row-major)")
        return arr.reshape(3, 3)
    raise ValueError(f"Unsupported orientation mode '{mode}'")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize and control the Inspire hand using FK/IK.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="Path to the hand URDF file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fk_parser = subparsers.add_parser("fk", help="Evaluate forward kinematics")
    fk_parser.add_argument(
        "--finger",
        choices=sorted(FINGER_TIP_LINKS.keys()),
        default="thumb",
        help="Finger to evaluate",
    )
    fk_parser.add_argument(
        "--joint",
        action="append",
        dest="joints",
        help="Override joint value (name=value in radians). Repeat as needed.",
    )
    fk_parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a 3D plot of the hand using the provided joint values",
    )

    ik_parser = subparsers.add_parser("ik", help="Solve inverse kinematics")
    ik_parser.add_argument(
        "--finger",
        choices=sorted(FINGER_TIP_LINKS.keys()),
        required=True,
        help="Finger to control",
    )
    ik_parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Desired fingertip position in meters (hand base frame)",
    )
    ik_parser.add_argument(
        "--orientation-mode",
        choices=["X", "Y", "Z", "all"],
        default=None,
        help="If set, also constrain fingertip orientation",
    )
    ik_parser.add_argument(
        "--orientation",
        nargs="+",
        type=float,
        help="Orientation vector/matrix (3 numbers for X/Y/Z modes, 9 numbers for 'all')",
    )
    ik_parser.add_argument(
        "--initial",
        action="append",
        dest="initial",
        help="Initial joint guess (name=value). Defaults to zero angles.",
    )
    ik_parser.add_argument(
        "--regularization",
        type=float,
        default=1e-3,
        help="Damping factor applied during IK optimization",
    )
    ik_parser.add_argument(
        "--position-tolerance",
        type=float,
        default=5e-4,
        help="Acceptable Cartesian error (meters) for IK feasibility",
    )
    ik_parser.add_argument(
        "--plot",
        action="store_true",
        help="Visualize the resulting IK pose",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)

    try:
        system = InspireHandKinematics(urdf_path=args.urdf)
    except Exception as exc:  # pylint: disable=broad-except
        parser.error(str(exc))

    if args.command == "fk":
        try:
            joint_state = parse_joint_overrides(args.joints)
        except ValueError as exc:
            parser.error(str(exc))
        transform = system.forward_kinematics(args.finger, joint_state)
        print(system.pretty_pose(transform))
        if args.plot:
            system.visualize(joint_state, highlight_finger=args.finger)
        else:
            print(f"Joint order for {args.finger}: {system.joint_order(args.finger)}")
        return 0

    if args.command == "ik":
        try:
            initial_state = parse_joint_overrides(args.initial)
        except ValueError as exc:
            parser.error(str(exc))
        try:
            orientation = parse_orientation(args.orientation, args.orientation_mode)
        except ValueError as exc:
            parser.error(str(exc))
        solution = system.inverse_kinematics(
            finger=args.finger,
            target_position=args.target,
            target_orientation=orientation,
            orientation_mode=args.orientation_mode,
            initial_state=initial_state,
            regularization=args.regularization,
            position_tolerance=args.position_tolerance,
        )
        print("Solved joint values (radians):")
        for name, value in solution.items():
            print(f"  {name}: {value:.5f}")
        if args.plot:
            joint_state = dict(initial_state)
            joint_state.update(solution)
            system.visualize(
                joint_state,
                highlight_finger=args.finger,
                target_points=[args.target],
            )
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
