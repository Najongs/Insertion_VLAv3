"""
Cache Status Checker

This script inspects the cache status for given dataset paths, checking for
the existence and dimensions of multiple cache types:
1.  CLIP 512d (from cache_clip_vlm_features.py)
2.  Raw 3B (from Make_VL_cache.py with a 3B model)
3.  Raw 7B (from Make_VL_cache.py with a 7B model)

It provides a summary of how many samples have each type of cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import glob
import os
import sys

# Add project root to import custom modules
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_cache_manager import VLACacheManager

# --- Start of Added Functions ---
# The following functions were added to resolve an ImportError, as they were not
# found in the specified import location. The logic is based on other scripts
# in the project like 'preview_clip_vlm_responses.py'.

CLIP_PROMPT_TEXT = "The following image shows a robot gripper, possibly with a tool, interacting with an object. Describe the visual state of the interaction area, focusing on the contact between the gripper/tool and the object. Key details to include are: the gripper's position relative to the target, the degree of insertion or contact, and any visible deformation or movement of the object. The task is to {task_name}."

def get_formatted_clip_prompt(task_name: str) -> str:
    """Formats the CLIP prompt with the given task name."""
    return CLIP_PROMPT_TEXT.format(task_name=task_name)

def get_clip_prompt_hash(task_name: str) -> str:
    """Generates a consistent hash for the formatted CLIP prompt."""
    prompt = get_formatted_clip_prompt(task_name)
    # Using 16 chars for uniqueness, as this seems to be the intended length for CLIP features.
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]

def extract_task_name_from_episode_path(episode_path: Path) -> str:
    """
    Extracts the task name from the episode path.
    e.g., /path/to/dataset/Red_point/episode_123 -> Red_point
    """
    # For both old and new formats, the task name is the parent directory of the episode.
    return episode_path.parent.name
# --- End of Added Functions ---

# Expected dimensions
DIM_512D = 512
DIM_RAW_3B = 2048
DIM_RAW_7B = 3584

def resolve_episode_dirs(raw_paths):
    """Expand glob patterns and return episode directories."""
    dirs = []
    print(f"DEBUG: resolve_episode_dirs called with raw_paths: {raw_paths}")
    for entry in raw_paths:
        expanded = []
        if any(ch in entry for ch in "*?[]"):
            expanded = sorted(Path(p) for p in glob.glob(entry))
            print(f"DEBUG: Glob pattern '{entry}' expanded to: {expanded}")
        else:
            expanded = [Path(entry)]
            print(f"DEBUG: Direct path entry: {entry}")

        if not expanded:
            print(f"DEBUG: No paths found for entry: {entry}")

        for path in expanded:
            path = path.expanduser().resolve()
            print(f"DEBUG: Processing path: {path}")
            if not path.exists():
                print(f"DEBUG: Path does not exist: {path}")
                continue
            if (path / "metadata.json").exists() or (path / "data.pkl").exists():
                print(f"DEBUG: Found episode directory: {path}")
                dirs.append(path)
                continue
            # Treat as root (task or dataset). Collect immediate children.
            print(f"DEBUG: Path is not a direct episode, checking subdirectories of {path}")
            for sub in sorted(path.iterdir()):
                if not sub.is_dir():
                    print(f"DEBUG: Skipping non-directory: {sub}")
                    continue
                if (sub / "metadata.json").exists() or (sub / "data.pkl").exists():
                    print(f"DEBUG: Found episode directory in subdirectory: {sub}")
                    dirs.append(sub)
    # Deduplicate while preserving order
    seen = {}
    for d in dirs:
        seen.setdefault(str(d), d)
    final_dirs = list(seen.values())
    print(f"DEBUG: Final resolved episode directories: {final_dirs}")
    return final_dirs

def get_instruction_from_metadata(episode_dir: Path) -> str:
    """Extracts instruction from metadata.json."""
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        # Fallback for old format might be needed, but for now, we assume new format
        # for raw features.
        task_name = extract_task_name_from_episode_path(episode_dir)
        return f"insert the {task_name.lower().replace('_', ' ')}."

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata.get("instruction", "")

def check_cache(
    episode_dirs: list[Path],
    cache_root: Path,
):
    """Checks the cache for each episode."""
    clip_cache_mgr = VLACacheManager(cache_dir=str(cache_root / "clip_vlm_features"))
    raw_cache_mgr = VLACacheManager(cache_dir=str(cache_root / "qwen_vl_features"))

    stats = {
        "512d": 0,
        "raw_3b": 0,
        "raw_7b": 0,
        "total_samples": 0,
    }

    print("🔍 Checking cache status for all episodes...")
    for episode_dir in tqdm(episode_dirs, desc="Episodes"):
        dataset_name = episode_dir.name
        
        # Determine number of samples
        metadata_path = episode_dir / "metadata.json"
        num_samples = 0 # Initialize num_samples
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            num_samples = metadata.get("num_samples", 0)
            print(f"DEBUG: Episode {dataset_name}: metadata.json exists, num_samples = {num_samples}")
        else: # old format
            print(f"DEBUG: Episode {dataset_name}: metadata.json does not exist. Trying old format.")
            try:
                # A bit of a hack: count images in View5 to estimate samples
                view5_dir = episode_dir / "images" / "View5"
                if not view5_dir.exists():
                    view5_dir = episode_dir / "View5"
                if view5_dir.exists():
                    num_samples = len(list(view5_dir.glob("*.jpg")))
                else:
                    num_samples = 0
                print(f"DEBUG: Episode {dataset_name}: Old format, num_samples = {num_samples}")
            except Exception as e:
                print(f"DEBUG: Episode {dataset_name}: Error counting images for old format: {e}")
                num_samples = 0

        if num_samples == 0:
            print(f"DEBUG: Skipping episode {dataset_name} due to 0 samples.")
            continue

        stats["total_samples"] += num_samples

        # Get prompt hashes
        task_name = extract_task_name_from_episode_path(episode_dir)
        clip_prompt_hash = get_clip_prompt_hash(task_name)
        
        instruction = get_instruction_from_metadata(episode_dir)
        raw_prompt_hash = hashlib.sha256(instruction.encode()).hexdigest()[:8]

        for vlm_idx in range(num_samples): # Check every potential sample
            # Check 512d cache
            if clip_cache_mgr.cache_exists(dataset_name, vlm_idx, clip_prompt_hash):
                stats["512d"] += 1

            # Check raw cache
            if raw_cache_mgr.cache_exists(dataset_name, vlm_idx, raw_prompt_hash):
                try:
                    cache_content = raw_cache_mgr.load_cache(dataset_name, vlm_idx, raw_prompt_hash)
                    if isinstance(cache_content, tuple) and len(cache_content) > 1:
                        guidance_vector = cache_content[1]
                        if isinstance(guidance_vector, torch.Tensor):
                            dim = guidance_vector.shape[-1]
                            if dim == DIM_RAW_3B:
                                stats["raw_3b"] += 1
                            elif dim == DIM_RAW_7B:
                                stats["raw_7b"] += 1
                except Exception as e:
                    print(f"Warning: Could not load raw cache for {dataset_name}_{vlm_idx}: {e}")


    return stats

def main():
    parser = argparse.ArgumentParser(description="Check dataset cache status.")
    parser.add_argument(
        '--dataset_paths', type=str, nargs='+', required=True,
        help='List of paths to the dataset directories (glob patterns allowed).'
    )
    parser.add_argument(
        '--cache_root', type=str, default='/home/najo/NAS/VLA/dataset/cache',
        help='Root directory for all caches.'
    )
    args = parser.parse_args()

    episode_dirs = resolve_episode_dirs(args.dataset_paths)
    if not episode_dirs:
        print("No valid episode directories found.")
        return

    stats = check_cache(episode_dirs, Path(args.cache_root))

    total = stats["total_samples"]
    if total == 0:
        print("\nNo samples found to check.")
        return

    print("\n--- Cache Status Summary ---")
    s_512 = stats['512d']
    s_3b = stats['raw_3b']
    s_7b = stats['raw_7b']
    
    print(f"Total samples checked: {total}")
    print("-" * 30)
    print(f"✓ CLIP 512d:     {s_512:>8,} / {total} ({s_512/total:.1%})")
    print(f"✓ Raw 3B (2048d):  {s_3b:>8,} / {total} ({s_3b/total:.1%})")
    print(f"✓ Raw 7B (3584d):  {s_7b:>8,} / {total} ({s_7b/total:.1%})")
    print("-" * 30)

    if s_7b > 0 and s_3b == 0:
        print("\nWarning: Found 'Raw 7B' cache but no 'Raw 3B' cache.")
        print("This might indicate that caching was accidentally run with a 7B model.")
        print(f"The 'Raw' cache is located in: {Path(args.cache_root) / 'qwen_vl_features'}")


if __name__ == "__main__":
    main()
