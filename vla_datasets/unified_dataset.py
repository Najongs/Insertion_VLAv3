"""
Unified VLA Dataset (통합 데이터셋)

두 가지 데이터 포맷을 지원하는 통합 데이터셋:
1. Old Format (AsyncInsertionMeca500DatasetWithSensor): data.pkl 기반
2. New Format (NewAsyncInsertionDataset): metadata.json + sensor_data.npz 기반

Key Features:
- VL feature caching support (Prompt-aware caching)
- Memory-optimized with mmap
- Weighted random sampling
- Async VLM update pattern (reuse_count)
"""

import glob
import hashlib
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torchvision import transforms
# Import VLA Cache Manager
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from vla_cache_manager import get_cache_manager

from .old_format_dataset import OldFormatDatasetMixin
from .new_format_dataset import NewFormatDatasetMixin


# =====================================
# Image Augmentation (Very Minimal)
# =====================================

class MinimalImageAugmentation:
    """
    Minimal conservative augmentation for VLA training.
    Only small rotation with 10% probability.
    """
    def __init__(self, prob: float = 0.10):
        self.prob = prob

    def __call__(self, image_path: str):
        """
        Args:
            image_path: Path to image file
        Returns:
            Augmented PIL Image or original if augmentation fails/skipped
        """
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")

            # Apply small rotation (10% probability)
            if np.random.random() < self.prob:
                angle = np.random.uniform(-2, 2)  # Very small angle: ±2 degrees
                img = transforms.functional.rotate(img, angle)

            return img
        except Exception as e:
            # If augmentation fails, return original image
            print(f"⚠️ Image augmentation failed for {image_path}: {e}")
            return Image.open(image_path).convert("RGB")


# =====================================
# Unified VLA Dataset
# =====================================

class UnifiedVLADataset(OldFormatDatasetMixin, NewFormatDatasetMixin, Dataset):
    """
    통합 VLA 데이터셋 - Old/New format 자동 감지

    Args:
        data_dir: 데이터 디렉토리 경로
        format: 'auto', 'old', 'new'
        horizon: Action prediction horizon (default: 8)
        vlm_reuse_count: VL feature reuse count (default: 3)
        sensor_window_size: Sensor window size (default: 65 for async, 650 for full)
        action_expert_hz: Action expert frequency (default: 10 Hz)
        cache_root: VL cache root directory
    """
    def __init__(
        self,
        data_dir: str,
        format: Literal['auto', 'old', 'new'] = 'auto',
        horizon: int = 8,
        vlm_reuse_count: int = 3,
        sensor_window_size: int = 65,
        robot_window_size: int = 100,
        action_expert_hz: int = 10,
        instruction: Optional[str] = None,
        cache_root: str = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
        use_cache: bool = True,
        use_augmentation: bool = False,
        augmentation_prob: float = 0.10,
        cache_build_only: bool = False,
        # Ablation study arguments
        views_to_use: Optional[List[int]] = None,
        disable_sensor: bool = False,
        disable_robot_state: bool = False,
        prompt_hash_override: Optional[str] = None,
        filter_by_cache: bool = False,
        verbose_cache_stats: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.use_cache = bool(use_cache)
        # Cache root is optional when cache is disabled (e.g., cache-build pipelines)
        if self.use_cache:
            if cache_root is None:
                raise ValueError("cache_root must be provided when use_cache=True")
            self.cache_root = Path(cache_root)
        else:
            self.cache_root = Path(cache_root) if cache_root else None
        self.horizon = int(horizon)
        self.vlm_reuse_count = int(vlm_reuse_count)
        self.sensor_window_size = int(sensor_window_size)
        self.robot_window_size = int(robot_window_size)
        self.action_expert_hz = int(action_expert_hz)
        self.sensor_hz = 650
        self.use_augmentation = use_augmentation
        self.cache_build_only = bool(cache_build_only)
        self.views_to_use = views_to_use
        self.disable_sensor = disable_sensor
        self.disable_robot_state = disable_robot_state
        self._prompt_hash_override = prompt_hash_override
        self._filter_by_cache = filter_by_cache
        self.verbose_cache_stats = bool(verbose_cache_stats)

        # Initialize augmentation (only if not using cache)
        if self.use_augmentation and not self.use_cache:
            self.augmentation = MinimalImageAugmentation(prob=augmentation_prob)
        else:
            self.augmentation = None

        # Auto-detect format
        if format == 'auto':
            format = self._detect_format()

        self.format = format

        from vla_cache_manager import get_cache_manager
        self.cache_mgr = get_cache_manager(cache_dir=str(self.cache_root)) if (self.use_cache and self.cache_root is not None) else None

        # Load data based on format
        if format == 'old':
            self._load_old_format(instruction)
        elif format == 'new':
            self._load_new_format(instruction)
        else:
            raise ValueError(f"Unknown format: {format}")

        # --- ✅ Generate prompt hash for versioned caching ---
        if self._prompt_hash_override:
            self.prompt_hash = self._prompt_hash_override
        else:
            self.prompt_hash = hashlib.sha256(self.instruction.encode()).hexdigest()[:8]
        
        # --- Apply Ablation Settings ---
        if self.disable_sensor:
            self.has_sensor = False
            # Replace sensor data with zeros to ensure consistent types
            if self.sensor_data is not None:
                self.sensor_data = np.zeros_like(self.sensor_data)

        if self.disable_robot_state:
            self.has_robot_states = False
            # Replace robot state data with zeros
            if self.robot_states is not None:
                self.robot_states = np.zeros_like(self.robot_states)

        if self.views_to_use is not None and self.images:
            filtered_images = {}
            view_keys = list(self.images.keys())
            for view_key in view_keys:
                # Check if any of the specified view numbers are in the view_key string
                if any(f"{v}" in view_key for v in self.views_to_use):
                    filtered_images[view_key] = self.images[view_key]
            self.images = filtered_images
            if not self.images:
                print(f"⚠️ WARNING: --views {self.views_to_use} resulted in no images being loaded for {self.data_dir.name}")
        # --- End of Ablation Settings ---

        # Pre-scan VL cache files (optimization)
        self._scan_vl_cache()

        # Filter out samples without cache if cache_only mode
        if self.use_cache and hasattr(self, '_filter_by_cache') and self._filter_by_cache:
            self._filter_samples_by_cache()

    def _detect_format(self) -> str:
        """Auto-detect dataset format"""
        if (self.data_dir / "metadata.json").exists():
            return 'new'
        elif (self.data_dir / "data.pkl").exists():
            return 'old'
        else:
            raise ValueError(f"Cannot detect format for {self.data_dir}")

    def _scan_vl_cache(self):
        """Pre-scan VL cache files using VLACacheManager with prompt hash."""
        self.vl_cache_files = {}
        dataset_name = self.data_dir.name
        found = 0

        if not self.use_cache or self.cache_mgr is None:
            self.vl_cache_files = {}
            self.cache_found_count = 0
            return

        if self.format == 'old':
            # Old format: vlm_idx based on action steps
            for action_step in range(self.max_action_steps):
                vlm_idx = min(action_step * self.action_step_size, len(self.actions) - 1)
                if vlm_idx in self.vl_cache_files:
                    continue

                cache_path = self.cache_mgr.get_cache_path(dataset_name, vlm_idx, self.prompt_hash)
                self.vl_cache_files[vlm_idx] = cache_path
                if cache_path.exists():
                    found += 1

        else:  # new format
            # New format: vlm_idx based on vlm_interval
            num_vlm_steps = (self._total_samples + self.vlm_reuse_count - 1) // self.vlm_reuse_count
            for i in range(num_vlm_steps):
                vlm_idx = i * self.vlm_interval
                cache_path = self.cache_mgr.get_cache_path(dataset_name, vlm_idx, self.prompt_hash)
                self.vl_cache_files[vlm_idx] = cache_path
                if cache_path.exists():
                    found += 1

        self.cache_found_count = found

    def _filter_samples_by_cache(self):
        """Filter out samples that don't have corresponding VL cache files."""
        if not self.use_cache or not self.vl_cache_files:
            return

        # Build a set of sample indices that have cache
        valid_indices = set()

        if self.format == 'old':
            # Old format: action_step based indexing
            for action_step in range(self.max_action_steps):
                vlm_idx = min(action_step * self.action_step_size, len(self.actions) - 1)
                cache_path = self.vl_cache_files.get(vlm_idx)
                if cache_path and cache_path.exists():
                    # All samples in this reuse group are valid
                    for reuse_step in range(self.vlm_reuse_count):
                        sample_idx = action_step * self.vlm_reuse_count + reuse_step
                        if sample_idx < self._total_samples:
                            valid_indices.add(sample_idx)
        else:
            # New format: vlm_interval based indexing
            for sample_idx in range(self._total_samples):
                vlm_group = sample_idx // self.vlm_reuse_count
                vlm_idx = vlm_group * self.vlm_interval
                cache_path = self.vl_cache_files.get(vlm_idx)
                if cache_path and cache_path.exists():
                    valid_indices.add(sample_idx)

        # Create index mapping
        self._valid_indices = sorted(list(valid_indices))
        original_count = self._total_samples
        self._total_samples = len(self._valid_indices)

        if self._total_samples < original_count and self.verbose_cache_stats:
            print(f"   🔍 {self.data_dir.name}: {self._total_samples}/{original_count} samples have cache ({self._total_samples/original_count*100:.1f}%)")

    def __len__(self):
        return self._total_samples

    def __getstate__(self):
        """Prepare for pickling - close file handles"""
        state = self.__dict__.copy()
        if self.format == 'new':
            state['sensor_npz'] = None
        state['cache_mgr'] = None
        return state

    def __setstate__(self, state):
        """Restore after unpickling"""
        self.__dict__.update(state)
        if self.format == 'new':
            self.sensor_npz = None
        from vla_cache_manager import get_cache_manager
        self.cache_mgr = get_cache_manager(cache_dir=str(self.cache_root))

    def __getitem__(self, idx):
        if idx >= self._total_samples:
            raise IndexError

        # Map to actual index if filtering by cache
        if hasattr(self, '_valid_indices'):
            idx = self._valid_indices[idx]

        if self.cache_build_only:
            return self._getitem_cache_only(idx)

        if self.format == 'old':
            return self._getitem_old(idx)
        else:
            return self._getitem_new(idx)

    def _getitem_cache_only(self, idx):
        reuse_step = idx % self.vlm_reuse_count
        if self.format == 'old':
            action_step = idx // self.vlm_reuse_count
            if len(self.actions) > 0:
                vlm_idx = min(action_step * self.action_step_size, len(self.actions) - 1)
            else:
                vlm_idx = 0
        else:
            action_step = idx
            vlm_idx = (idx // self.vlm_reuse_count) * self.vlm_interval if self.vlm_reuse_count > 0 else 0

        _, image_paths = self._load_vl_or_images(vlm_idx)
        cache_key = f"{self.data_dir.name}_vlm{vlm_idx}"

        sample = {
            "instruction": self.instruction,
            "images": image_paths,
            "vl_cache": None,
            "sensor_data": torch.zeros((1, 1026), dtype=torch.float32),
            "robot_states": torch.zeros((1, 12), dtype=torch.float32),
            "actions": torch.zeros((self.horizon, 7), dtype=torch.float32),
            "has_sensor": False,
            "has_robot_states": False,
            "cache_key": cache_key,
            "vlm_idx": int(vlm_idx),
            "reuse_step": int(reuse_step),
            "confidence": 0.0,
            "prompt_hash": self.prompt_hash,
        }

        if self.format != 'old':
            timestamp = 0.0
            if image_paths:
                try:
                    timestamp = float(Path(image_paths[0]).stem)
                except (ValueError, IndexError):
                    timestamp = 0.0
            sample["episode_id"] = self.data_dir.name
            sample["timestamp"] = timestamp

        return sample

    def _load_vl_or_images(self, vlm_idx):
        """Load VL cache or return image paths/PIL images using VLACacheManager"""
        vl_cache = None
        image_data = []

        cache_path = self.vl_cache_files.get(vlm_idx)

        if self.use_cache and cache_path:
            # Use cache manager for loading
            vl_cache = self.cache_mgr.load_cache(
                dataset_name=self.data_dir.name,
                vlm_idx=vlm_idx,
                prompt_hash=self.prompt_hash,
                device="cpu"
            )

            # Track cache statistics (class-level to share across all instances)
            if not hasattr(self.__class__, '_cache_stats'):
                self.__class__._cache_stats = {'miss': 0, 'hit': 0, 'total': 0}

            if vl_cache is None:
                self.__class__._cache_stats['miss'] += 1
                self.__class__._cache_stats['total'] += 1

                # Print summary every 100 accesses
                if self.verbose_cache_stats and self.__class__._cache_stats['total'] % 100 == 1:
                    miss_rate = (self.__class__._cache_stats['miss'] / self.__class__._cache_stats['total']) * 100
                    print(f"📊 Cache: {self.__class__._cache_stats['hit']} hits / {self.__class__._cache_stats['miss']} misses ({miss_rate:.1f}% miss rate)")
            else:
                self.__class__._cache_stats['hit'] += 1
                self.__class__._cache_stats['total'] += 1

            if vl_cache is not None:
                return vl_cache, []

        # Fallback to image paths (with optional augmentation)
        if isinstance(self.images, dict):
            for view_name in sorted(self.images.keys()):
                view_images = self.images[view_name]
                if len(view_images) > 0:
                    img_idx = min(vlm_idx, len(view_images) - 1)
                    if self.format == 'old':
                        img_path = view_images[img_idx]
                        img_path_str = img_path if img_path else ""
                    else:
                        img_path_str = view_images[img_idx]

                    # Apply augmentation if enabled (only when not using cache)
                    if self.augmentation is not None and img_path_str:
                        try:
                            augmented_img = self.augmentation(img_path_str)
                            image_data.append(augmented_img)
                        except Exception as e:
                            print(f"⚠️ Augmentation failed for {img_path_str}: {e}")
                            image_data.append(img_path_str)
                    else:
                        image_data.append(img_path_str)

        return vl_cache, image_data


# =====================================
# Collate Function
# =====================================

def unified_collate_fn(batch):
    """
    통합 collate function for batching

    Returns:
        Dictionary with batched data
    """
    instructions = [b["instruction"] for b in batch]
    
    # Sanitize image lists to prevent errors with None values
    image_lists_raw = [b.get("images") for b in batch]
    image_lists = []
    for sublist in image_lists_raw:
        if sublist is None:
            image_lists.append([])  # Replace top-level None with an empty list
        else:
            # Filter out any None paths within the sublist
            sanitized_sublist = [item for item in sublist if item is not None]
            image_lists.append(sanitized_sublist)

    vl_features = [b["vl_cache"] for b in batch]

    # Pad sensor data to max length
    sensor_tensors = [b["sensor_data"] for b in batch]
    max_sensor_len = max(t.shape[0] for t in sensor_tensors) if sensor_tensors else 0
    padded_sensors = []
    for sensor in sensor_tensors:
        if sensor.shape[0] < max_sensor_len:
            pad = torch.zeros((max_sensor_len - sensor.shape[0], sensor.shape[1]),
                            dtype=sensor.dtype)
            padded_sensors.append(torch.cat([sensor, pad], dim=0))
        else:
            padded_sensors.append(sensor)
    sensor_data = torch.stack(padded_sensors, dim=0) if padded_sensors else torch.empty(0)

    # Pad robot states to max length (same as sensor data)
    robot_state_tensors = [b["robot_states"] for b in batch]
    max_robot_len = max(t.shape[0] for t in robot_state_tensors) if robot_state_tensors else 0
    padded_robot_states = []
    for robot_state in robot_state_tensors:
        if robot_state.shape[0] < max_robot_len:
            pad = torch.zeros((max_robot_len - robot_state.shape[0], robot_state.shape[1]),
                            dtype=robot_state.dtype)
            padded_robot_states.append(torch.cat([robot_state, pad], dim=0))
        else:
            padded_robot_states.append(robot_state)
    robot_states = torch.stack(padded_robot_states, dim=0) if padded_robot_states else torch.empty(0)

    actions = torch.stack([b["actions"] for b in batch], dim=0)
    has_sensor_mask = torch.tensor([b["has_sensor"] for b in batch], dtype=torch.bool)
    has_robot_states_mask = torch.tensor([b["has_robot_states"] for b in batch], dtype=torch.bool)
    cache_keys = [b["cache_key"] for b in batch]
    vlm_indices = [b["vlm_idx"] for b in batch]
    reuse_steps = [b["reuse_step"] for b in batch]
    confidence = [b["confidence"] for b in batch]
    prompt_hashes = [b.get("prompt_hash") for b in batch]
    episode_ids = [b.get("episode_id") for b in batch]

    return {
        "instruction": instructions,
        "images": image_lists,
        "vl_cache": vl_features,
        "sensor_data": sensor_data,
        "robot_states": robot_states,
        "actions": actions,
        "has_sensor_mask": has_sensor_mask,
        "has_robot_states_mask": has_robot_states_mask,
        "cache_keys": cache_keys,
        "vlm_indices": vlm_indices,
        "reuse_steps": reuse_steps,
        "confidence": confidence,
        "prompt_hash": prompt_hashes,
        "episode_ids": episode_ids,
    }


# =====================================
# Unified Dataloader Builder
# =====================================

def create_unified_dataloader(
    old_dataset_patterns: List[str] = None,
    new_dataset_path: Optional[str] = None,
    new_dataset_paths: Optional[List[str]] = None,
    dataset_weights: Optional[List[str]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    horizon: int = 8,
    vlm_reuse_count: int = 3,
    sensor_window_size: int = 65,
    robot_window_size: int = 100,
    action_expert_hz: int = 10,
    cache_root: str = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    use_cache: bool = True,
    use_augmentation: bool = False,
    augmentation_prob: float = 0.10,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    return_dataset: bool = False,
    # Ablation study arguments
    views_to_use: Optional[List[int]] = None,
    disable_sensor: bool = False,
    disable_robot_state: bool = False,
    cache_build_only: bool = False,
    prompt_hash_override: Optional[str] = None,
    skip_dataset_stats: bool = False,
    filter_by_cache: bool = False,
):
    """
    통합 데이터로더 생성

    Args:
        ... (기존 인자들)
        dataset_weights: A list of strings in 'path:weight' format.
        views_to_use: 사용할 View 번호 리스트
        disable_sensor: 센서 데이터 비활성화 여부
        disable_robot_state: 로봇 상태 데이터 비활성화 여부
        skip_dataset_stats: 데이터셋 통계 수집/출력 생략 (빠른 시작)
    """
    datasets = []
    weights_for_sampler = []
    track_weights = (not distributed) and shuffle and not skip_dataset_stats
    old_sample_count = 0
    new_sample_count = 0
    cache_hit_count = 0
    cache_slot_count = 0

    # Parse dataset_weights
    def _has_wildcard_pattern(path_str: str) -> bool:
        return any(ch in path_str for ch in "*?[]")

    weight_map = {}
    if dataset_weights:
        raw_weight_map = {}
        for item in dataset_weights:
            path, weight = item.split(':', 1)
            raw_weight_map[path] = float(weight)

        for path, weight in raw_weight_map.items():
            if _has_wildcard_pattern(path):
                matched_paths = glob.glob(path)
                if matched_paths:
                    for matched in matched_paths:
                        weight_map[matched] = weight
                else:
                    # Keep original entry so we can warn later when it fails to resolve
                    weight_map[path] = weight
            else:
                weight_map[path] = weight

    # Common dataset args
    dataset_kwargs = {
        "horizon": horizon,
        "vlm_reuse_count": vlm_reuse_count,
        "sensor_window_size": sensor_window_size,
        "robot_window_size": robot_window_size,
        "action_expert_hz": action_expert_hz,
        "cache_root": cache_root,
        "use_cache": use_cache,
        "use_augmentation": use_augmentation,
        "augmentation_prob": augmentation_prob,
        "views_to_use": views_to_use,
        "disable_sensor": disable_sensor,
        "disable_robot_state": disable_robot_state,
        "cache_build_only": cache_build_only,
        "prompt_hash_override": prompt_hash_override,
        "filter_by_cache": filter_by_cache,
        "verbose_cache_stats": not skip_dataset_stats,
    }

    # Load old format datasets
    if old_dataset_patterns:
        for pattern in old_dataset_patterns:
            expanded_paths = sorted(glob.glob(pattern))
            for traj_dir in expanded_paths:
                try:
                    ds = UnifiedVLADataset(
                        data_dir=traj_dir,
                        format='old',
                        **dataset_kwargs
                    )
                    datasets.append(ds)
                    if not skip_dataset_stats:
                        old_sample_count += len(ds)
                        if use_cache and hasattr(ds, "vl_cache_files"):
                            cache_slot_count += len(ds.vl_cache_files)
                            cache_hit_count += getattr(ds, "cache_found_count", 0)
                    if track_weights:
                        weight = weight_map.get(traj_dir, 1.0)
                        weights_for_sampler.extend([weight] * len(ds))
                except Exception as e:
                    print(f"⚠️ Failed to load old dataset {traj_dir}: {e}")

    # Load new format datasets
    new_paths_to_process = []
    if new_dataset_path:
        new_paths_to_process.append(new_dataset_path)
    if new_dataset_paths:
        new_paths_to_process.extend(new_dataset_paths)

    if new_paths_to_process:
        expanded_new_paths = []
        for raw_entry in new_paths_to_process:
            entry_str = str(raw_entry)
            if _has_wildcard_pattern(entry_str):
                matched = sorted(glob.glob(entry_str))
                if matched:
                    if rank == 0 and not skip_dataset_stats:
                        print(f"   - Pattern '{entry_str}' expanded to {len(matched)} paths.")
                    expanded_new_paths.extend(matched)
                    continue
            expanded_new_paths.append(entry_str)
        new_paths_to_process = expanded_new_paths

    if new_paths_to_process:
        for new_dataset_path_item in new_paths_to_process:
            new_path = Path(new_dataset_path_item)
            if not new_path.exists():
                if rank == 0:
                    print(f"   - Path not found, skipping: {new_path}")
                continue

            path_weight = weight_map.get(str(new_path))

            # Determine if new_path is a root of tasks or a single task directory
            subdirs = sorted([d for d in new_path.iterdir() if d.is_dir()])
            is_task_dir = any(d.name.startswith('episode_') or d.name.startswith('data_collection_') for d in subdirs)

            if is_task_dir:
                # It's a task directory. The subdirs are episodes.
                task_list = [new_path]
            else:
                # It's a root directory. The subdirs are tasks.
                task_list = subdirs

            for task_dir in task_list:
                for episode_dir in sorted(task_dir.iterdir()):
                    if not episode_dir.is_dir() or not (episode_dir.name.startswith('episode_') or episode_dir.name.startswith('data_collection_')):
                        continue

                    try:
                        if rank == 0 and not cache_build_only and not skip_dataset_stats:
                            print(f"     - Loading new format: {episode_dir}")
                        ds = UnifiedVLADataset(
                            data_dir=str(episode_dir),
                            format='new',
                            **dataset_kwargs
                        )
                        if rank == 0 and not skip_dataset_stats:
                            print(f"       - Loaded dataset {episode_dir} with {len(ds)} samples.")
                        if len(ds) > 0:
                            datasets.append(ds)
                            if not skip_dataset_stats:
                                new_sample_count += len(ds)
                                if use_cache:
                                    cache_slot_count += len(ds.vl_cache_files)
                                    cache_hit_count += getattr(ds, "cache_found_count", 0)
                            if track_weights:
                                # Prioritize episode-specific weight, then path-specific, then default
                                weight = weight_map.get(str(episode_dir), path_weight if path_weight is not None else 1.0)
                                weights_for_sampler.extend([weight] * len(ds))
                    except Exception as e:
                        print(f"⚠️ Failed to load new dataset {episode_dir}: {e}")

    if not datasets:
        raise ValueError("No datasets loaded! Please check dataset paths and formats.")

    full_dataset = ConcatDataset(datasets)

    if not skip_dataset_stats:
        print(f"\n📊 Total dataset statistics:")
        print(f"   Total samples: {len(full_dataset)}")
        print(f"   Total datasets: {len(datasets)}")
        print(f"   Old dataset samples: {old_sample_count}")
        print(f"   New dataset samples: {new_sample_count}")

        if track_weights and weights_for_sampler:
            # Create a summary of weights
            weight_summary = {}
            for path, weight in weight_map.items():
                # Find which datasets match this path
                matching_samples = 0
                for ds in datasets:
                    if str(ds.data_dir).startswith(path):
                        matching_samples += len(ds)
                if matching_samples > 0:
                    weight_summary[path] = {'weight': weight, 'samples': matching_samples}

            if weight_summary:
                print("   Weighted sampling enabled:")
                for path, info in weight_summary.items():
                    print(f"     - Path: {path}, Weight: {info['weight']}, Samples: {info['samples']}")

        if use_cache and cache_slot_count > 0:
            cache_ratio = cache_hit_count / cache_slot_count
            print(f"   VL cache coverage: {cache_hit_count}/{cache_slot_count} ({cache_ratio:.2%})")
    else:
        print(f"⚡ Dataset statistics skipped (--skip_dataset_stats enabled). Loaded {len(datasets)} datasets.")

    full_dataset.cache_hit_count = cache_hit_count
    full_dataset.cache_slot_count = cache_slot_count

    if return_dataset:
        full_dataset.num_old_samples = old_sample_count
        full_dataset.num_new_samples = new_sample_count
        return full_dataset

    # Create sampler
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            full_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False
    elif track_weights and weights_for_sampler:
        sampler = WeightedRandomSampler(
            weights=weights_for_sampler,
            num_samples=len(weights_for_sampler),
            replacement=True,
        )
        shuffle = False

    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=unified_collate_fn,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        pin_memory_device='cuda' if num_workers > 0 else '',
    )

    return dataloader


# Backward compatibility aliases
AsyncInsertionMeca500DatasetWithSensor = lambda **kwargs: UnifiedVLADataset(format='old', **kwargs)
NewAsyncInsertionDataset = lambda **kwargs: UnifiedVLADataset(format='new', **kwargs)
async_collate_fn_with_sensor = unified_collate_fn
create_weighted_async_dataloader = create_unified_dataloader


# =====================================
# Test Code
# =====================================

if __name__ == "__main__":
    print("🧪 Testing Unified VLA Dataset...")

    # Test old format
    print("\n=== Testing Old Format ===")
    old_test_dir = "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_170308"
    if Path(old_test_dir).exists():
        try:
            ds_old = UnifiedVLADataset(
                data_dir=old_test_dir,
                format='old',
                horizon=8,
                vlm_reuse_count=3,
                sensor_window_size=65,
            )
            print(f"✅ Old dataset loaded: {len(ds_old)} samples")

            if len(ds_old) > 0:
                sample = ds_old[0]
                print(f"   Instruction: {sample['instruction']}")
                print(f"   Prompt Hash: {sample['prompt_hash']}")
                print(f"   Sensor shape: {sample['sensor_data'].shape}")
                print(f"   Actions shape: {sample['actions'].shape}")
                print(f"   Has sensor: {sample['has_sensor']}")
        except Exception as e:
            print(f"⚠️ Old format test failed: {e}")
    else:
        print(f"⚠️ Old test directory not found: {old_test_dir}")

    # Test new format
    print("\n=== Testing New Format ===")
    new_test_dir = "/home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Blue_point/episode_20251030_025119"
    if Path(new_test_dir).exists():
        try:
            ds_new = UnifiedVLADataset(
                data_dir=new_test_dir,
                format='new',
                horizon=8,
                vlm_reuse_count=3,
                sensor_window_size=650,
            )
            print(f"✅ New dataset loaded: {len(ds_new)} samples")

            if len(ds_new) > 0:
                sample = ds_new[0]
                print(f"   Instruction: {sample['instruction']}")
                print(f"   Prompt Hash: {sample['prompt_hash']}")
                print(f"   Sensor shape: {sample['sensor_data'].shape}")
                print(f"   Actions shape: {sample['actions'].shape}")
                print(f"   Has sensor: {sample['has_sensor']}")
        except Exception as e:
            print(f"⚠️ New format test failed: {e}")
    else:
        print(f"⚠️ New test directory not found: {new_test_dir}")

    print("\n✅ All tests completed!")
