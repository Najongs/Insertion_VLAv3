"""
VLA Cache Manager - Prompt-Aware Caching System

Cache File Name Rule:
- Path: {cache_dir}/{prompt_hash}/{dataset_name}_vlm{vlm_idx}.pt
- The `prompt_hash` isolates caches based on the instruction content,
  preventing conflicts when prompts are changed.

Example:
- /cache/a1b2c3d4/recv_all_20251027_170308_vlm0.pt
- /cache/e5f6g7h8/episode_20251030_025119_vlm0.pt
"""

import hashlib
import fcntl
import os
from pathlib import Path
from typing import Optional, List
import torch


class VLACacheManager:
    """
    VLA Cache Manager - Prompt-Aware Caching

    Features:
    - Creates versioned cache paths using a hash of the prompt.
    - Atomic save for safe concurrent access.
    - Manages disk space with a cache limit.
    """

    def __init__(
        self,
        cache_dir: str = "/dev/shm/vla_cache",
        cache_limit_gb: float = 50.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_limit_gb = cache_limit_gb

    def _raw_cache_path(self, dataset_name: str, vlm_idx: int, prompt_hash: str) -> Path:
        return (self.cache_dir / prompt_hash) / f"{dataset_name}_vlm{vlm_idx}.pt"

    def get_cache_path(
        self,
        dataset_name: str,
        vlm_idx: int,
        prompt_hash: str,
    ) -> Path:
        """
        Generate a prompt-aware cache file path, ensuring the directory exists.
        """
        versioned_dir = self.cache_dir / prompt_hash
        versioned_dir.mkdir(parents=True, exist_ok=True)
        return versioned_dir / f"{dataset_name}_vlm{vlm_idx}.pt"

    def cache_exists(
        self,
        dataset_name: str,
        vlm_idx: int,
        prompt_hash: str,
    ) -> bool:
        """Check if a cache file exists for a given prompt hash."""
        cache_path = self._raw_cache_path(dataset_name, vlm_idx, prompt_hash)
        return cache_path.exists()

    def load_cache(
        self,
        dataset_name: str,
        vlm_idx: int,
        prompt_hash: str,
        device: str = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        Load cache for a specific prompt hash.

        Returns:
            Cached VL features tensor or None if not found.
        """
        cache_path = self._raw_cache_path(dataset_name, vlm_idx, prompt_hash)

        if not cache_path.exists():
            return None

        try:
            cached = torch.load(cache_path, map_location=device)
            return cached
        except Exception as e:
            print(f"⚠️ Failed to load cache {cache_path.name}: {e}")
            return None

    def save_cache(
        self,
        dataset_name: str,
        vlm_idx: int,
        prompt_hash: str,
        vl_features: (torch.Tensor | dict),
    ):
        """
        Save cache atomically. Handles single tensors or dicts of tensors.
        """
        cache_path = self.get_cache_path(dataset_name, vlm_idx, prompt_hash)

        data_to_save = None
        if isinstance(vl_features, dict):
            data_to_save = {
                k: v.detach().to("cpu", dtype=torch.float16)
                for k, v in vl_features.items() if isinstance(v, torch.Tensor)
            }
        elif isinstance(vl_features, torch.Tensor):
            data_to_save = vl_features.detach().to("cpu", dtype=torch.float16)
        else:
            # For other types that can be pickled directly
            data_to_save = vl_features

        if data_to_save is None:
            return

        # Atomic save with file lock
        self._atomic_save(data_to_save, cache_path)

        # Enforce cache limit
        self._enforce_cache_limit()

    def save_cache_tuple(
        self,
        dataset_name: str,
        vlm_idx: int,
        prompt_hash: str,
        features_tuple: tuple,
    ):
        """
        Save cache as a tuple (e.g., image_features, guidance_vector).
        Each element in the tuple should already be detached and on CPU.
        """
        cache_path = self.get_cache_path(dataset_name, vlm_idx, prompt_hash)

        # Atomic save with file lock
        self._atomic_save(features_tuple, cache_path)

        # Enforce cache limit
        self._enforce_cache_limit()

    @staticmethod
    def _atomic_save(tensor_cpu: torch.Tensor, path: Path):
        """Atomic save with file lock"""
        tmp = path.with_suffix(".pt.tmp")
        lock_path = str(path) + ".lock"

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lockfile:
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX)

                if path.exists():
                    return

                torch.save(tensor_cpu, tmp)
                os.replace(tmp, path)

            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)
                # Clean up lock file
                try:
                    os.remove(lock_path)
                except OSError:
                    pass

    def _enforce_cache_limit(self):
        """Apply cache size limit across all versioned subdirectories."""
        if self.cache_limit_gb <= 0:
            return

        all_files = list(self.cache_dir.rglob("*.pt"))
        total_bytes = sum(f.stat().st_size for f in all_files)
        limit_bytes = self.cache_limit_gb * (1024 ** 3)

        if total_bytes <= limit_bytes:
            return

        # Sort files by modification time (oldest first)
        all_files.sort(key=lambda f: f.stat().st_mtime)

        for file in all_files:
            if total_bytes <= limit_bytes:
                break
            try:
                size = file.stat().st_size
                file.unlink(missing_ok=True)
                total_bytes -= size
            except FileNotFoundError:
                continue

    def get_cache_stats(self) -> dict:
        """Get statistics for the entire cache."""
        all_files = list(self.cache_dir.rglob("*.pt"))
        total_bytes = sum(f.stat().st_size for f in all_files)
        total_gb = total_bytes / (1024 ** 3)

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(all_files),
            "total_size_gb": total_gb,
            "limit_gb": self.cache_limit_gb,
            "usage_percent": (total_gb / self.cache_limit_gb * 100) if self.cache_limit_gb > 0 else 0,
        }

    def clear_cache(self, confirm: bool = False):
        """Clear the entire cache, including all subdirectories."""
        if not confirm:
            print("⚠️ Cache clear requires confirm=True")
            return

        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Cleared all cache files and subdirectories from {self.cache_dir}")

    def list_cached_datasets(self) -> dict:
        """List cached datasets grouped by prompt_hash."""
        all_files = list(self.cache_dir.rglob("*.pt"))
        
        versioned_datasets = {}
        for f in all_files:
            try:
                prompt_hash = f.parent.name
                name = f.stem
                if "_vlm" in name:
                    dataset_name, vlm_part = name.rsplit("_vlm", 1)
                    vlm_idx = int(vlm_part)

                    if prompt_hash not in versioned_datasets:
                        versioned_datasets[prompt_hash] = {}
                    if dataset_name not in versioned_datasets[prompt_hash]:
                        versioned_datasets[prompt_hash][dataset_name] = []
                    
                    versioned_datasets[prompt_hash][dataset_name].append(vlm_idx)
            except (ValueError, IndexError):
                continue

        # Sort vlm indices
        for prompt_hash, datasets in versioned_datasets.items():
            for dataset_name in datasets:
                datasets[dataset_name].sort()

        return versioned_datasets


# Global cache manager instance
_cache_manager = None


def get_cache_manager(
    cache_dir: str = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    cache_limit_gb: float = 50.0,
) -> VLACacheManager:
    """Get global cache manager instance"""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = VLACacheManager(
            cache_dir=cache_dir,
            cache_limit_gb=cache_limit_gb,
        )
    # Allow re-configuration if needed
    elif _cache_manager.cache_dir != Path(cache_dir):
         _cache_manager = VLACacheManager(
            cache_dir=cache_dir,
            cache_limit_gb=cache_limit_gb,
        )

    return _cache_manager


if __name__ == "__main__":
    print("🧪 Testing VLA Cache Manager (Prompt-Aware)...")

    # Create cache manager
    test_cache_dir = Path("/tmp/test_vla_cache_prompt_aware")
    if test_cache_dir.exists():
        import shutil
        shutil.rmtree(test_cache_dir)

    cache_mgr = VLACacheManager(
        cache_dir=str(test_cache_dir),
        cache_limit_gb=1.0,
    )

    # Test cache path generation
    print("\n📁 Cache path generation:")
    prompt1_hash = "a1b2c3d4"
    prompt2_hash = "e5f6g7h8"
    path1 = cache_mgr.get_cache_path("recv_all_data", 0, prompt1_hash)
    path2 = cache_mgr.get_cache_path("episode_data", 150, prompt2_hash)
    print(f"   Path 1: {path1}")
    print(f"   Path 2: {path2}")
    assert path1 == test_cache_dir / prompt1_hash / "recv_all_data_vlm0.pt"
    assert path2 == test_cache_dir / prompt2_hash / "episode_data_vlm150.pt"

    # Test save and load
    print("\n💾 Save and load test:")
    test_features = torch.randn(1, 1, 3072)
    cache_mgr.save_cache("test_dataset", 0, prompt1_hash, test_features)
    print(f"   Saved to: {cache_mgr.get_cache_path('test_dataset', 0, prompt1_hash)}")

    loaded = cache_mgr.load_cache("test_dataset", 0, prompt1_hash)
    assert loaded is not None, "Failed to load cache"
    print(f"   Loaded: {loaded.shape}")
    print(f"   Match: {torch.allclose(test_features.cpu().float(), loaded.float(), rtol=1e-3)}")

    # Test loading non-existent cache
    loaded_none = cache_mgr.load_cache("test_dataset", 0, prompt2_hash)
    assert loaded_none is None, "Should not find cache for different prompt hash"
    print("   ✅ Correctly returned None for non-existent cache.")

    # Test stats
    print("\n📊 Cache statistics:")
    stats = cache_mgr.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    assert stats['total_files'] == 1

    # Test list datasets
    print("\n📋 Cached datasets:")
    datasets = cache_mgr.list_cached_datasets()
    print(datasets)
    assert prompt1_hash in datasets
    assert "test_dataset" in datasets[prompt1_hash]

    # Cleanup
    cache_mgr.clear_cache(confirm=True)
    assert not test_cache_dir.exists()

    print("\n✅ All tests passed!")
