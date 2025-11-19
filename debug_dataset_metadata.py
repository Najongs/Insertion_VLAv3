#!/usr/bin/env python3
"""
Quick script to debug dataset metadata (episode_ids, vlm_indices, prompt_hash)
"""
import sys
sys.path.append('/home/najo/NAS/VLA/Insertion_VLAv3')

from vla_datasets.unified_dataset import create_unified_dataloader

# Test with one dataset
dataset_paths = ["/home/najo/NAS/VLA/dataset/New_dataset6/Yellow_point"]

print("🔍 Testing dataset metadata...")
print(f"Dataset: {dataset_paths[0]}\n")

dataloader = create_unified_dataloader(
    new_dataset_paths=dataset_paths,
    batch_size=4,
    num_workers=0,  # Single process for debugging
    shuffle=False,
    horizon=10,
    vlm_reuse_count=1,
    cache_root="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    use_cache=True,
    distributed=False,
    skip_dataset_stats=True,
)

print("\n" + "="*80)
print("📦 Getting first batch...")
print("="*80 + "\n")

batch = next(iter(dataloader))

print("Keys in batch:", list(batch.keys()))
print()

# Check metadata fields
for key in ["episode_ids", "vlm_indices", "prompt_hash"]:
    value = batch.get(key)
    print(f"{key}:")
    print(f"  Type: {type(value)}")
    if value is not None:
        print(f"  Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
        print(f"  Sample: {value[:2] if hasattr(value, '__getitem__') else value}")
    else:
        print(f"  ⚠️ VALUE IS NONE!")
    print()

print("="*80)
print("✅ Metadata check complete")
