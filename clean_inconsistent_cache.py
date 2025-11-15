#!/usr/bin/env python3
"""
Clean inconsistent CLIP VLM cache files.

This script identifies and optionally removes cache files that have
different feature dimensions, helping resolve dimension mismatch errors.
"""

import torch
from pathlib import Path
from collections import defaultdict
import argparse


def analyze_cache_dimensions(cache_root: Path):
    """Analyze all cache files and group by dimension."""
    dimension_groups = defaultdict(list)

    for pt_file in cache_root.rglob("*.pt"):
        try:
            data = torch.load(pt_file, map_location='cpu')

            if isinstance(data, tuple) and len(data) >= 2:
                img_feat, guid_vec = data[0], data[1]
                img_dim = img_feat.shape[-1]
                guid_dim = guid_vec.shape[-1]

                if img_dim == guid_dim:
                    dimension_groups[img_dim].append(pt_file)
                else:
                    print(f"⚠️ Mismatched dims in single file: {pt_file.name} (img={img_dim}, guid={guid_dim})")

        except Exception as e:
            print(f"⚠️ Error reading {pt_file.name}: {e}")

    return dimension_groups


def main():
    parser = argparse.ArgumentParser(description="Clean inconsistent CLIP VLM cache files")
    parser.add_argument(
        "--cache-root",
        type=str,
        default="/home/najo/NAS/VLA/dataset/cache/clip_vlm_features",
        help="Path to CLIP VLM cache root directory"
    )
    parser.add_argument(
        "--remove-minority",
        action="store_true",
        help="Remove files with minority dimension (use with caution!)"
    )
    parser.add_argument(
        "--keep-dimension",
        type=int,
        default=None,
        help="Keep only files with this specific dimension"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing"
    )

    args = parser.parse_args()
    cache_root = Path(args.cache_root)

    if not cache_root.exists():
        print(f"❌ Cache root not found: {cache_root}")
        return

    print(f"🔍 Analyzing cache files in: {cache_root}")
    dimension_groups = analyze_cache_dimensions(cache_root)

    # Print summary
    print(f"\n📊 Cache Dimension Summary:")
    print("=" * 60)
    total_files = sum(len(files) for files in dimension_groups.values())

    for dim, files in sorted(dimension_groups.items(), key=lambda x: len(x[1]), reverse=True):
        percentage = (len(files) / total_files * 100) if total_files > 0 else 0
        print(f"  Dimension {dim:4d}: {len(files):6d} files ({percentage:5.1f}%)")

    print("=" * 60)
    print(f"  Total:          {total_files:6d} files")

    # Determine what to remove
    files_to_remove = []

    if args.keep_dimension is not None:
        # Remove all except specified dimension
        for dim, files in dimension_groups.items():
            if dim != args.keep_dimension:
                files_to_remove.extend(files)

        if files_to_remove:
            print(f"\n🎯 Will keep only dimension {args.keep_dimension}")
            print(f"   Removing {len(files_to_remove)} files with other dimensions")

    elif args.remove_minority:
        # Remove dimension with fewer files
        if len(dimension_groups) > 1:
            sorted_dims = sorted(dimension_groups.items(), key=lambda x: len(x[1]))
            majority_dim, majority_files = sorted_dims[-1]

            for dim, files in sorted_dims[:-1]:
                files_to_remove.extend(files)

            print(f"\n🎯 Majority dimension: {majority_dim} ({len(majority_files)} files)")
            print(f"   Removing {len(files_to_remove)} minority files")

    # Remove files
    if files_to_remove:
        if args.dry_run:
            print(f"\n🔍 DRY RUN: Would remove {len(files_to_remove)} files:")
            for f in files_to_remove[:10]:  # Show first 10
                print(f"     {f.relative_to(cache_root)}")
            if len(files_to_remove) > 10:
                print(f"     ... and {len(files_to_remove) - 10} more")
        else:
            print(f"\n🗑️  Removing {len(files_to_remove)} files...")
            removed_count = 0

            for f in files_to_remove:
                try:
                    f.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to remove {f}: {e}")

            print(f"✅ Successfully removed {removed_count} files")
    else:
        print("\n✅ No inconsistencies found or no action specified.")
        print("   All cache files have consistent dimensions!")

    # Final recommendation
    if len(dimension_groups) > 1 and not files_to_remove:
        print("\n💡 Recommendation:")
        print("   Use --remove-minority or --keep-dimension to clean up inconsistent cache")
        print("   Example: python clean_inconsistent_cache.py --remove-minority --dry-run")


if __name__ == "__main__":
    main()
