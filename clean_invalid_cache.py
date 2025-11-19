#!/usr/bin/env python3
"""
Clean invalid VL cache files (where image features are empty)
"""
import torch
from pathlib import Path
from tqdm import tqdm

def is_cache_invalid(cache_path):
    """Check if a cache file has empty image features"""
    try:
        cached = torch.load(cache_path, map_location='cpu')

        # Check if it's a tuple (image_features, text_features)
        if isinstance(cached, tuple) and len(cached) == 2:
            image_features, text_features = cached

            # Check if image_features is a tensor and has empty sequence dimension
            if isinstance(image_features, torch.Tensor):
                if len(image_features.shape) >= 2 and image_features.shape[1] == 0:
                    return True  # Invalid: empty image features

        return False  # Valid cache

    except Exception as e:
        print(f"⚠️ Error loading {cache_path.name}: {e}")
        return True  # Consider corrupt files as invalid

def clean_invalid_caches(cache_root, dry_run=True):
    """
    Clean all invalid cache files in the cache directory.

    Args:
        cache_root: Root cache directory
        dry_run: If True, only print what would be deleted without actually deleting
    """
    cache_root = Path(cache_root)

    if not cache_root.exists():
        print(f"❌ Cache directory not found: {cache_root}")
        return

    # Find all .pt cache files
    all_cache_files = list(cache_root.rglob("*.pt"))
    print(f"📊 Found {len(all_cache_files)} cache files to check")

    invalid_files = []
    valid_files = []

    # Check each cache file
    for cache_file in tqdm(all_cache_files, desc="Checking cache files"):
        if is_cache_invalid(cache_file):
            invalid_files.append(cache_file)
        else:
            valid_files.append(cache_file)

    # Report findings
    print(f"\n📈 Results:")
    print(f"   ✅ Valid caches: {len(valid_files)}")
    print(f"   ❌ Invalid caches: {len(invalid_files)}")

    if invalid_files:
        if dry_run:
            print(f"\n🔍 DRY RUN MODE - Would delete {len(invalid_files)} invalid cache files:")
            for i, f in enumerate(invalid_files[:10]):  # Show first 10
                print(f"      {f.relative_to(cache_root)}")
            if len(invalid_files) > 10:
                print(f"      ... and {len(invalid_files) - 10} more")
            print(f"\n💡 Run with --delete to actually delete these files")
        else:
            print(f"\n🗑️  Deleting {len(invalid_files)} invalid cache files...")
            deleted_count = 0
            for cache_file in tqdm(invalid_files, desc="Deleting"):
                try:
                    cache_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {cache_file.name}: {e}")

            print(f"✅ Successfully deleted {deleted_count}/{len(invalid_files)} invalid cache files")

            # Calculate freed space
            total_size = sum(f.stat().st_size for f in cache_root.rglob("*.pt"))
            print(f"📦 Remaining cache size: {total_size / (1024**3):.2f} GB")
    else:
        print("\n✨ All cache files are valid!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean invalid VL cache files")
    parser.add_argument(
        "--cache_root",
        type=str,
        default="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
        help="Root cache directory"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete invalid files (default: dry run)"
    )

    args = parser.parse_args()

    clean_invalid_caches(
        cache_root=args.cache_root,
        dry_run=not args.delete
    )
