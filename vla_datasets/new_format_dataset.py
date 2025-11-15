"""
New-format dataset mixin extracted from unified dataset implementation.

Handles metadata.json + npz based trajectories.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation


class NewFormatDatasetMixin:
    """Mixin containing logic specific to metadata/json based trajectories."""

    def _load_new_format(self, instruction: Optional[str]):
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.robot_hz = self.meta.get("robot_hz", 100)
        self.sensor_hz = self.meta.get("sensor_hz", 650)
        self.action_interval = int(self.robot_hz / self.action_expert_hz)
        self.vlm_interval = self.action_interval * self.vlm_reuse_count

        # =============================================================================
        # ⚠️ CRITICAL: Instruction 생성과 VL 캐시 매칭
        # =============================================================================
        # 이 instruction 텍스트는 SHA256 해시화되어 prompt_hash를 생성합니다.
        # VL 캐시는 다음 경로에 저장됩니다:
        #   {cache_root}/{prompt_hash}/{episode_name}_vlm{idx}.pt
        #
        # 문제점:
        #   - task_name (예: "Red point", "Blue point")이 instruction에 포함됨
        #   - 태스크마다 다른 instruction → 다른 prompt_hash
        #   - 따라서 태스크별로 별도의 캐시 디렉토리 생성됨
        #
        # 예시:
        #   Red_point → "...target is the Red point..." → prompt_hash: a1b2c3d4
        #   Blue_point → "...target is the Blue point..." → prompt_hash: e5f6g7h8
        #   캐시 경로: /cache/a1b2c3d4/, /cache/e5f6g7h8/
        #
        # 해결 방법:
        #   1. 모든 태스크의 캐시를 생성 (현재 방법)
        #   2. prompt_hash_override로 모든 태스크가 같은 캐시 사용 (instruction 무시)
        #   3. task_name을 instruction에서 제거하여 통일된 prompt_hash 사용
        #
        # 현재 상태:
        #   - 캐시 생성 시 vlm_reuse_count=3 사용
        #   - 학습 시에도 동일한 vlm_reuse_count 필요
        #   - prompt_hash_override 없이 자동 생성된 hash 사용
        # =============================================================================

        task_name = self.data_dir.parent.name.replace("_", " ")

        if instruction is None:
            self.instruction = f"""Respond ONLY with the next action.
Environment Context:
- This is a Meca500 robot workspace.
- The end-effector holds a needle; the needle tip is the tool.
- The scene is an optical table with many holes, but these are NOT targets.
- The ONLY true insertion target is the {task_name}.

Task:
You must analyze the five camera views and determine the needle’s relative position to the {task_name}.
Identify:
1) needle tip location
2) alignment relative to the {task_name} center
3) required direction to align for insertion

Respond with:
- target visibility
- needle alignment
- required adjustment direction
- insertion readiness (yes/no)
"""
        else:
            self.instruction = instruction

        # Find timestamped sensor file: sensor_data_YYYYMMDD_HHMMSS.npz
        sensor_files = list(self.data_dir.glob("sensor_data_*.npz"))
        if sensor_files:
            # Use the first timestamped sensor file found
            self.sensor_path = sensor_files[0]
        else:
            # Fallback to sensor_data.npz if no timestamped file exists
            self.sensor_path = self.data_dir / "sensor_data.npz"

        self.sensor_npz = None
        self.sensor_raw_data = None  # Will store raw alines and forces
        self._load_sensor_metadata()

        npz_path = self.data_dir / "robot_states.npz"
        csv_path = self.data_dir / "robot_states.csv"

        if npz_path.exists():
            try:
                with np.load(npz_path, mmap_mode="r") as data:
                    self.robot_states = np.array(data["robot_states"], dtype=np.float32)
                    self.joints = (
                        np.array(data["joints"], dtype=np.float32)
                        if "joints" in data
                        else self.robot_states[:, :6]
                    )
                    self.poses = (
                        np.array(data["poses"], dtype=np.float32)
                        if "poses" in data
                        else self.robot_states[:, 6:]
                    )
                self.num_poses = len(self.poses)
                self.has_robot_states = True
            except Exception:
                self.robot_states = np.zeros((1, 12), dtype=np.float32)
                self.joints = np.zeros((1, 6), dtype=np.float32)
                self.poses = np.zeros((1, 6), dtype=np.float32)
                self.num_poses = 1
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
            except Exception:
                df = pd.read_csv(csv_path)

            self.joints = df[joint_cols].to_numpy(dtype=np.float32)
            self.poses = df[pose_cols].to_numpy(dtype=np.float32)
            self.num_poses = len(self.poses)
            self.robot_states = np.concatenate([self.joints, self.poses], axis=1)
            self.has_robot_states = True
        else:
            self.robot_states = np.zeros((1, 12), dtype=np.float32)
            self.joints = np.zeros((1, 6), dtype=np.float32)
            self.poses = np.zeros((1, 6), dtype=np.float32)
            self.num_poses = 1
            self.has_robot_states = False

        self.actions = None

        # Check for images in both "images/" subdirectory and directly in data_dir
        self.images = {}
        for view_name in self.meta.get("camera_views", []):
            # First try: images/ViewX directory
            img_dir = self.data_dir / "images"
            view_dir = img_dir / view_name
            if view_dir.exists():
                files = sorted(view_dir.glob("*.jpg"), key=lambda x: self._extract_timestamp(x.stem))
                self.images[view_name] = [str(f) for f in files]
            else:
                # Second try: ViewX directory directly in data_dir
                view_dir = self.data_dir / view_name
                if view_dir.exists():
                    files = sorted(view_dir.glob("*.jpg"), key=lambda x: self._extract_timestamp(x.stem))
                    self.images[view_name] = [str(f) for f in files]

        self.num_actions = max(0, (self.num_poses - self.action_interval) // self.action_interval)
        self._total_samples = self.num_actions
        self.action_step_size = self.action_interval
        self.max_action_steps = self.num_actions

    # Remaining helpers are reused verbatim from the original implementation.

    def _extract_timestamp(self, filename: str) -> float:
        """
        Extract timestamp from image filename.
        Supports formats:
        - Simple numeric: "1762725445.926" -> 1762725445.926
        - With prefix: "ZED_41182735_left_1762725445.926" -> 1762725445.926
        """
        try:
            # Try direct conversion first
            return float(filename)
        except ValueError:
            # Extract last numeric part (timestamp)
            parts = filename.split('_')
            for part in reversed(parts):
                try:
                    return float(part)
                except ValueError:
                    continue
            # Fallback: return 0
            return 0.0

    def _load_sensor_metadata(self):
        """Load raw sensor data (alines + forces) from timestamped npz file."""
        try:
            with np.load(self.sensor_path) as npz:
                # Check if this is raw data (has 'alines' and 'forces') or preprocessed data (has 'data')
                if 'alines' in npz and 'forces' in npz:
                    # Raw sensor data format
                    self.sensor_timestamps = npz["timestamps"][:]
                    alines = npz["alines"]  # Shape: (N, 1025)
                    forces = npz["forces"]  # Shape: (N,)

                    # Store shapes for reference
                    self.sensor_length = len(self.sensor_timestamps)
                    self.alines_shape = alines.shape
                    self.forces_shape = forces.shape
                    self.has_sensor = True

                    # Don't load all data into memory yet - will be loaded on-demand
                    # Just store the file path for lazy loading

                elif 'data' in npz:
                    # Fallback: preprocessed window data
                    print(f"⚠️ Using preprocessed sensor data (not raw). Consider regenerating.")
                    self.sensor_timestamps = npz["timestamps"][:]
                    self.sensor_windows_shape = npz["data"].shape
                    self.sensor_length = self.sensor_windows_shape[0]
                    self.has_sensor = True
                else:
                    raise ValueError(f"Unknown sensor data format in {self.sensor_path}")

        except FileNotFoundError:
            print(f"⚠️ Sensor file not found: {self.sensor_path}")
            self.sensor_timestamps = np.array([])
            self.sensor_length = 0
            self.has_sensor = False
        except Exception as e:
            print(f"⚠️ Error loading sensor metadata {self.sensor_path}: {e}")
            self.sensor_length = 0
            self.has_sensor = False

    def _get_sensor_npz(self):
        if self.sensor_npz is None or not hasattr(self.sensor_npz, "f"):
            try:
                self.sensor_npz = np.load(self.sensor_path, mmap_mode="r")
            except FileNotFoundError:
                return None
        return self.sensor_npz

    def _getitem_new(self, idx: int):
        reuse_step = idx % self.vlm_reuse_count
        action_step = idx
        vlm_idx = (idx // self.vlm_reuse_count) * self.vlm_interval

        vl_cache, image_paths = self._load_vl_or_images(vlm_idx)
        sensor_window = self._get_sensor_window_new(idx)
        robot_state_window = self._get_robot_state_window_new(idx)
        actions = self._get_actions_new(action_step)

        cache_key = f"{self.data_dir.name}_vlm{vlm_idx}"

        timestamp = 0.0
        if image_paths and image_paths[0]:
            try:
                timestamp = float(Path(image_paths[0]).stem)
            except (ValueError, IndexError):
                pass

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
            "confidence": 1.0,
            "episode_id": self.data_dir.name,
            "timestamp": timestamp,
            "prompt_hash": self.prompt_hash,
        }

    def _get_sensor_window_new(self, idx: int):
        """
        Extract sensor window from raw data and combine alines + forces.

        Returns:
            np.ndarray: Shape (sensor_window_size, 1026) where 1026 = 1025 (alines) + 1 (forces)
        """
        if not self.has_sensor:
            return np.zeros((self.sensor_window_size, 1026), dtype=np.float32)

        s_npz = self._get_sensor_npz()
        if s_npz is None or self.sensor_length == 0:
            return np.zeros((self.sensor_window_size, 1026), dtype=np.float32)

        # Check if using raw data or preprocessed data
        if 'alines' in s_npz and 'forces' in s_npz:
            # Raw sensor data: extract window and combine alines + forces

            # Calculate sensor index corresponding to robot action index
            # robot_idx = idx * action_interval (robot states at 100Hz)
            # sensor samples at 650Hz, so sensor_idx ≈ robot_idx * (sensor_hz / robot_hz)
            robot_center_idx = idx * self.action_interval
            sensor_ratio = self.sensor_hz / self.robot_hz  # ~6.5
            sensor_center_idx = int(robot_center_idx * sensor_ratio)

            # Extract window centered around sensor_center_idx
            sensor_start = max(0, sensor_center_idx - self.sensor_window_size // 2)
            sensor_end = sensor_start + self.sensor_window_size

            # Ensure we don't exceed data bounds
            sensor_end = min(sensor_end, self.sensor_length)
            sensor_start = max(0, sensor_end - self.sensor_window_size)

            # Load data slice
            alines_window = np.array(s_npz["alines"][sensor_start:sensor_end], dtype=np.float32)  # (N, 1025)
            forces_window = np.array(s_npz["forces"][sensor_start:sensor_end], dtype=np.float32)  # (N,)

            # Combine: alines (1025) + forces (1) = 1026 dimensions
            forces_expanded = forces_window[:, np.newaxis]  # (N, 1)
            sensor_window = np.concatenate([alines_window, forces_expanded], axis=1)  # (N, 1026)

            # Pad if necessary
            if sensor_window.shape[0] < self.sensor_window_size:
                pad = np.zeros((self.sensor_window_size - sensor_window.shape[0], 1026), dtype=np.float32)
                sensor_window = np.concatenate([sensor_window, pad], axis=0)

            return sensor_window

        elif 'data' in s_npz:
            # Fallback: preprocessed window data
            sensor_idx = min(idx, self.sensor_length - 1)
            sensor_window = np.array(s_npz["data"][sensor_idx], dtype=np.float32)

            if sensor_window.shape[0] < self.sensor_window_size:
                pad = np.zeros((self.sensor_window_size - sensor_window.shape[0], sensor_window.shape[1]), dtype=np.float32)
                sensor_window = np.concatenate([sensor_window, pad], axis=0)

            return sensor_window

        else:
            return np.zeros((self.sensor_window_size, 1026), dtype=np.float32)

    def _get_robot_state_window_new(self, idx: int):
        if not self.has_robot_states:
            return np.zeros((self.robot_window_size, 12), dtype=np.float32)

        center_idx = idx * self.action_interval
        start_idx = max(0, center_idx - self.robot_window_size // 2)
        end_idx = start_idx + self.robot_window_size

        end_idx = min(end_idx, len(self.robot_states))
        rw = self.robot_states[start_idx:end_idx]

        if rw.shape[0] < self.robot_window_size:
            pad = np.zeros((self.robot_window_size - rw.shape[0], 12), dtype=np.float32)
            return np.concatenate([rw, pad], axis=0)

        return rw

    def _get_actions_new(self, action_step: int):
        actions = []

        for i in range(self.horizon):
            current_action_idx = action_step + i
            start_pose_idx = current_action_idx * self.action_interval
            end_pose_idx = start_pose_idx + self.action_interval

            if end_pose_idx >= self.num_poses:
                break

            if (self.num_actions - current_action_idx) <= 5:
                delta_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            else:
                start_pose = self.poses[start_pose_idx]
                end_pose = self.poses[end_pose_idx]

                delta_translation = end_pose[:3] - start_pose[:3]
                r_start = Rotation.from_euler("xyz", start_pose[3:], degrees=True)
                r_end = Rotation.from_euler("xyz", end_pose[3:], degrees=True)
                r_delta = r_end * r_start.inv()
                delta_rotation = r_delta.as_rotvec()

                delta_pose = np.concatenate([delta_translation, delta_rotation])
                delta_action = np.concatenate([delta_pose, [1.0]], axis=0)

            actions.append(delta_action)

        if not actions:
            default_action = np.array([0.0] * 6 + [1.0], dtype=np.float32)
            return np.tile(default_action, (self.horizon, 1))
        if len(actions) < self.horizon:
            pad = np.tile(actions[-1], (self.horizon - len(actions), 1))
            actions = np.concatenate([actions, pad], axis=0)

        return np.array(actions, dtype=np.float32)


# =====================================
# Test Code
# =====================================

if __name__ == "__main__":
    print("🧪 Testing NewFormatDatasetMixin...")
    print()

    # We need to create a minimal dataset class that uses the mixin
    # since the mixin requires the UnifiedVLADataset infrastructure

    from pathlib import Path
    import sys

    # Add parent directory to path to import UnifiedVLADataset
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from vla_datasets.unified_dataset import UnifiedVLADataset

    # Test with New_dataset3
    test_episode = "/home/najo/NAS/VLA/dataset/New_dataset2/Green_point/data_collection_20251108_053848"

    if not Path(test_episode).exists():
        print(f"❌ Test episode not found: {test_episode}")
        sys.exit(1)

    print(f"📂 Testing with: {test_episode}")
    print()

    try:
        # Create dataset instance
        print("1️⃣ Creating dataset instance...")
        ds = UnifiedVLADataset(
            data_dir=test_episode,
            format='new',
            horizon=8,
            vlm_reuse_count=3,
            sensor_window_size=650,
            robot_window_size=100,
            action_expert_hz=10,
            use_cache=False,  # Disable cache for testing
        )
        print(f"   ✅ Dataset created successfully")
        print(f"   Total samples: {len(ds)}")
        print()

        # Check metadata
        print("2️⃣ Checking metadata...")
        print(f"   Format: {ds.format}")
        print(f"   Robot Hz: {ds.robot_hz}")
        print(f"   Sensor Hz: {ds.sensor_hz}")
        print(f"   Action interval: {ds.action_interval}")
        print(f"   VLM interval: {ds.vlm_interval}")
        print(f"   Horizon: {ds.horizon}")
        print(f"   Has sensor: {ds.has_sensor}")
        print(f"   Has robot states: {ds.has_robot_states}")
        print()

        # Check data shapes
        print("3️⃣ Checking data shapes...")
        if ds.has_sensor:
            print(f"   Sensor data shape: {ds.sensor_windows_shape}")
            print(f"   Sensor length: {ds.sensor_length}")
        if ds.has_robot_states:
            print(f"   Robot states shape: {ds.robot_states.shape}")
            print(f"   Joints shape: {ds.joints.shape}")
            print(f"   Poses shape: {ds.poses.shape}")
        print(f"   Number of actions: {ds.num_actions}")
        print()

        # Check camera views
        print("4️⃣ Checking camera views...")
        for view_name, view_images in ds.images.items():
            print(f"   {view_name}: {len(view_images)} images")
        print()

        # Test getting a sample
        print("5️⃣ Testing sample retrieval...")
        if len(ds) > 0:
            sample = ds[0]
            print(f"   Sample 0 retrieved successfully")
            print(f"   Instruction length: {len(sample['instruction'])} chars")
            print(f"   Images: {len(sample['images'])} views")
            print(f"   Sensor data shape: {sample['sensor_data'].shape}")
            print(f"   Robot states shape: {sample['robot_states'].shape}")
            print(f"   Actions shape: {sample['actions'].shape}")
            print(f"   Cache key: {sample['cache_key']}")
            print(f"   VLM idx: {sample['vlm_idx']}")
            print(f"   Reuse step: {sample['reuse_step']}")
            print(f"   Episode ID: {sample['episode_id']}")
            print(f"   Timestamp: {sample['timestamp']}")
            print()

            # Test middle sample
            if len(ds) > 10:
                mid_idx = len(ds) // 2
                sample_mid = ds[mid_idx]
                print(f"   Sample {mid_idx} retrieved successfully")
                print(f"   Sensor data shape: {sample_mid['sensor_data'].shape}")
                print(f"   Actions shape: {sample_mid['actions'].shape}")
                print()
        else:
            print("   ⚠️ Dataset is empty!")
            print()

        # Test action computation
        print("6️⃣ Testing action computation...")
        if ds.num_actions > 0:
            actions = ds._get_actions_new(0)
            print(f"   Actions shape: {actions.shape}")
            print(f"   Action range: [{actions[:, :6].min():.4f}, {actions[:, :6].max():.4f}]")
            print(f"   Gripper values: {actions[:, 6]}")

            # Check if last actions are stop actions
            if ds.num_actions > 5:
                last_actions = ds._get_actions_new(ds.num_actions - 3)
                print(f"   Near-end actions (should have stop signals):")
                print(f"   Action magnitudes: {np.linalg.norm(last_actions[:, :6], axis=1)}")
        print()

        print("=" * 70)
        print("✅ All tests passed!")
        print()
        print("Summary:")
        print(f"  ✅ Dataset loads from new format (metadata.json)")
        print(f"  ✅ Sensor data: {ds.sensor_windows_shape}")
        print(f"  ✅ Robot states: {ds.robot_states.shape}")
        print(f"  ✅ Images: {sum(len(v) for v in ds.images.values())} total")
        print(f"  ✅ Samples: {len(ds)}")
        print()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
