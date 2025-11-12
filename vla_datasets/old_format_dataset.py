"""
Old-format dataset mixin extracted from unified dataset implementation.

Provides data.pkl-based loading along with helper utilities that are only
needed for the legacy trajectories.
"""

import gc
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


def _safe_joblib_load(pkl_path: Path):
    """Try joblib (mmap) first, then fall back to pickle."""
    try:
        import joblib

        return joblib.load(pkl_path, mmap_mode="r")
    except Exception:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)


def _load_sensor_compound(traj_dir: Path, T_actions: int):
    """Load sensor data from .npy or from data.pkl fallback."""
    npy_path = traj_dir / "sensor_data.npy"
    if npy_path.exists():
        arr = np.load(npy_path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape == (T_actions, 1026):
            return arr

    data = _safe_joblib_load(traj_dir / "data.pkl")
    sensor_raw = data.get("sensor_data")
    if sensor_raw is None:
        return np.zeros((T_actions, 1026), dtype=np.float32)

    if isinstance(sensor_raw, dict):
        fpi_data = sensor_raw.get("fpi", np.zeros((T_actions, 1025), dtype=np.float32))
        force_data = sensor_raw.get("force", np.zeros((T_actions, 1), dtype=np.float32))
        return np.column_stack((force_data, fpi_data)).astype(np.float32, copy=False)

    return np.asarray(sensor_raw, dtype=np.float32)


class OldFormatDatasetMixin:
    """Mixin containing logic specific to legacy data.pkl trajectories."""

    def _load_old_format(self, instruction: Optional[str]):
        data_file = self.data_dir / "data.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"data.pkl not found in {self.data_dir}")

        data = _safe_joblib_load(data_file)

        actions = data.get("action")
        if actions is None:
            raise ValueError(f"'action' missing in {data_file}")
        self.actions = np.asarray(actions, dtype=np.float32)
        T = len(self.actions)

        self.images = data.get("image", {}) or {}

        if instruction is None:
            base_instruction = data.get("instruction", "needle insertion")
            task_name = base_instruction.replace("Perform ", "").replace(" task.", "")

            if self.use_cache:
                view_names = list(self.images.keys()) if self.images else []
                if view_names:
                    view_descriptions = []
                    for view_name in view_names:
                        if "5" in str(view_name) or "oak" in str(view_name).lower():
                            view_descriptions.append(
                                f"[{view_name}] HAND-EYE CAMERA - CRITICAL: IDENTIFY the insertion target. "
                                "LOCATE its exact position. TRACK it continuously."
                            )
                        elif "1" in str(view_name) or "front" in str(view_name).lower():
                            view_descriptions.append(
                                f"[{view_name}] FRONT VIEW: Locate the target and plan approach trajectory."
                            )
                        elif "2" in str(view_name) or "side" in str(view_name).lower():
                            view_descriptions.append(
                                f"[{view_name}] SIDE VIEW: Determine depth and check alignment with target."
                            )
                        else:
                            view_descriptions.append(f"[{view_name}] Analyze spatial relationships.")
                    view_guide = "\n".join(view_descriptions)
                    self.instruction = (
                        f"ROBOTICS VISION TASK: {task_name}\n\n"
                        f"MULTI-VIEW ANALYSIS:\n{view_guide}\n\n"
                        f"OBJECTIVE: Generate robot actions to insert into the identified target location."
                    )
                else:
                    self.instruction = (
                        f"ROBOTICS VISION TASK: {task_name}. "
                        "OBJECTIVE: Generate robot actions to accurately insert into the identified target."
                    )
            else:
                self.instruction = (
                    "You are an expert robot operator for a delicate insertion task. "
                    f"Your goal is to guide the robot to insert its tool into the '{task_name}' target."
                    "Analyze the image and determine the next action."
                    "Output your analysis in this format: "
                    "1) Target Analysis: [FULLY_VISIBLE/PARTIALLY_VISIBLE/NOT_VISIBLE], [FAR/MID/NEAR/TOUCHING]. "
                    "2) Current State: [Briefly describe the tool-target relationship]. "
                    "3) Next Action: [Choose ONE: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, "
                    "MOVE_UP, MOVE_DOWN, ROTATE_CW, ROTATE_CCW, ALIGN_TARGET, INSERT, STOP]. "
                    "4) Confidence: [HIGH/MEDIUM/LOW]."
                )
        else:
            self.instruction = instruction

        self.has_sensor = ("sensor_data" in data) and (data["sensor_data"] is not None)
        if self.has_sensor:
            npy_path = self.data_dir / "sensor_data.npy"
            if npy_path.exists():
                self.sensor_data = np.load(npy_path, mmap_mode="r")
                if not (self.sensor_data.ndim == 2 and self.sensor_data.shape == (T, 1026)):
                    self.sensor_data = _load_sensor_compound(self.data_dir, T)
            else:
                self.sensor_data = _load_sensor_compound(self.data_dir, T)
        else:
            self.sensor_data = np.zeros((T, 1026), dtype=np.float32)

        npz_path = self.data_dir / "robot_states.npz"
        csv_path = self.data_dir / "robot_states.csv"

        if npz_path.exists():
            try:
                with np.load(npz_path, mmap_mode="r") as npz_data:
                    self.robot_states = np.array(npz_data["robot_states"], dtype=np.float32)
                    self.joints = (
                        np.array(npz_data["joints"], dtype=np.float32)
                        if "joints" in npz_data
                        else self.robot_states[:, :6]
                    )
                    self.poses = (
                        np.array(npz_data["poses"], dtype=np.float32)
                        if "poses" in npz_data
                        else self.robot_states[:, 6:]
                    )
                self.has_robot_states = True
            except Exception:
                self.robot_states = np.zeros((T, 12), dtype=np.float32)
                self.has_robot_states = False
        elif csv_path.exists():
            joint_cols = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            pose_cols = ["pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"]
            use_cols = joint_cols + pose_cols
            try:
                if not hasattr(self.__class__, "_csv_warning_shown"):
                    print("   ⚠️ Loading robot states from CSV (slow). Consider converting to NPZ.")
                    self.__class__._csv_warning_shown = True
                df = pd.read_csv(csv_path, usecols=use_cols)
                self.joints = df[joint_cols].to_numpy(dtype=np.float32)
                self.poses = df[pose_cols].to_numpy(dtype=np.float32)
                self.robot_states = np.concatenate([self.joints, self.poses], axis=1)
                self.has_robot_states = True
            except Exception:
                self.robot_states = np.zeros((T, 12), dtype=np.float32)
                self.has_robot_states = False
        else:
            self.robot_states = np.zeros((T, 12), dtype=np.float32)
            self.has_robot_states = False

        del data
        gc.collect()

        self.action_step_size = 1
        self.max_action_steps = max(0, T - self.horizon)
        self._total_samples = self.max_action_steps

        self.robot_hz = 100
        self.action_interval = int(self.robot_hz / self.action_expert_hz)

    def _getitem_old(self, idx: int):
        action_step = idx
        reuse_step = idx % self.vlm_reuse_count
        vlm_idx_base = (idx // self.vlm_reuse_count) * self.vlm_reuse_count
        vlm_idx = min(vlm_idx_base, len(self.actions) - 1)

        vl_cache, image_paths = self._load_vl_or_images(vlm_idx)

        sensor_start = action_step
        sensor_end = sensor_start + self.sensor_window_size
        sensor_window = self._get_sensor_window_old(sensor_start, sensor_end)

        robot_state_window = self._get_robot_state_window_old(sensor_start, sensor_end)

        action_start = action_step
        action_end = action_start + self.horizon
        actions = self._get_actions_old(action_start, action_end)

        cache_key = f"{self.data_dir.name}_vlm{vlm_idx}"

        return {
            "instruction": self.instruction,
            "images": image_paths,
            "vl_cache": vl_cache,
            "sensor_data": torch.from_numpy(sensor_window),
            "robot_states": torch.from_numpy(robot_state_window),
            "actions": torch.from_numpy(actions),
            "has_sensor": bool(self.has_sensor),
            "has_robot_states": bool(self.has_robot_states),
            "cache_key": cache_key,
            "vlm_idx": int(vlm_idx),
            "reuse_step": int(reuse_step),
            "confidence": 1.0 if self.has_sensor else 0.5,
            "prompt_hash": self.prompt_hash,
        }

    def _get_sensor_window_old(self, start: int, end: int):
        T_sensor = len(self.sensor_data) if hasattr(self.sensor_data, "__len__") else 0

        if T_sensor == 0 or start >= T_sensor:
            return np.zeros((self.sensor_window_size, 1026), dtype=np.float32)

        sw = self.sensor_data[start : min(end, T_sensor)]
        if sw.shape[0] < self.sensor_window_size:
            pad = np.zeros((self.sensor_window_size - sw.shape[0], 1026), dtype=np.float32)
            return np.concatenate([sw, pad], axis=0)
        return sw

    def _get_robot_state_window_old(self, start: int, end: int):
        if not self.has_robot_states:
            return np.zeros((self.robot_window_size, 12), dtype=np.float32)

        T_robot = len(self.robot_states)
        if T_robot == 0 or start >= T_robot:
            return np.zeros((self.robot_window_size, 12), dtype=np.float32)

        robot_end = min(start + self.robot_window_size, T_robot)
        rw = self.robot_states[start:robot_end]

        if rw.shape[0] < self.robot_window_size:
            pad = np.zeros((self.robot_window_size - rw.shape[0], 12), dtype=np.float32)
            return np.concatenate([rw, pad], axis=0)
        return rw

    def _get_actions_old(self, start: int, end: int):
        T_action = len(self.actions)

        if start >= T_action:
            return np.zeros((self.horizon, 7), dtype=np.float32)

        act = self.actions[start : min(end, T_action)].copy()

        for i in range(len(act)):
            current_frame_idx = start + i
            if (T_action - current_frame_idx) <= 5:
                act[i, :6] = 0.0

        if act.shape[0] < self.horizon:
            last = act[-1] if act.shape[0] > 0 else np.zeros((7,), dtype=np.float32)
            pad = np.tile(last, (self.horizon - act.shape[0], 1))
            return np.concatenate([act, pad], axis=0)
        return act

