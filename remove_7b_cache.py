"""
Remove 7B Cache Files

This script scans a specified cache directory for 'raw' VLA cache files,
identifies those created by a 7B model (dimension 3584), and deletes them.
"""
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import os

# The dimension of the guidance vector from the 7B model
DIM_RAW_7B = 3584

def scan_and_delete_7b_cache(cache_dir: Path, dry_run: bool = True):
    """
    Scans for and deletes cache files matching the 7B model dimension.

    Args:
        cache_dir: The path to the 'qwen_vl_features' directory.
        dry_run: If True, only prints the files that would be deleted.
                 If False, performs the deletion.
    """
    if not cache_dir.exists():
        print(f"Error: Cache directory does not exist: {cache_dir}")
        return

    print(f"Scanning for 7B cache files (dim={DIM_RAW_7B}) in: {cache_dir}")
    
    # Use rglob to find all .pt files recursively
    files_to_delete = []
    all_pt_files = list(cache_dir.rglob("*.pt"))

    if not all_pt_files:
        print("No .pt files found in the directory.")
        return

    for i, file_path in enumerate(tqdm(all_pt_files, desc="Checking files")):
        try:
            cache_content = torch.load(file_path, map_location='cpu')
            
            # --- DEBUG: Print structure of first 10 files ---
            if i < 10:
                print(f"\n--- Inspecting file #{i+1}: {file_path} ---")
                if isinstance(cache_content, tuple):
                    print(f"Type: tuple, Length: {len(cache_content)}")
                    for j, item in enumerate(cache_content):
                        if isinstance(item, torch.Tensor):
                            print(f"  Item {j}: Tensor, Shape: {item.shape}, DType: {item.dtype}")
                        else:
                            print(f"  Item {j}: Type: {type(item)}")
                else:
                    print(f"Type: {type(cache_content)}")
            # --- END DEBUG ---

            # The raw cache is a tuple: (image_features, guidance_vector)
            if not isinstance(cache_content, tuple) or len(cache_content) < 2:
                continue

            guidance_vector = cache_content[1]
            if not isinstance(guidance_vector, torch.Tensor):
                continue

            # Check the last dimension of the guidance vector
            if guidance_vector.shape[-1] == DIM_RAW_7B:
                files_to_delete.append(file_path)

        except Exception as e:
            print(f"\nWarning: Could not process file {file_path}: {e}")
            continue

    if not files_to_delete:
        print("\nNo 7B cache files found to delete.")
        return

    print(f"\nFound {len(files_to_delete)} cache files from a 7B model.")

    if dry_run:
        print("--- DRY RUN ---")
        print("The following files would be deleted:")
        for f in files_to_delete:
            print(f"  - {f}")
        print("\nTo delete these files, run the script again with the --delete flag.")
    else:
        print("--- DELETING FILES ---")
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"Deleted: {f}")
            except OSError as e:
                print(f"Error deleting {f}: {e}")
        print(f"\nDeletion complete. {len(files_to_delete)} files removed.")

def main():
    parser = argparse.ArgumentParser(description="Remove 7B VLA cache files.")
    parser.add_argument(
        '--cache_dir', 
        type=str, 
        default='/home/najo/NAS/VLA/dataset/cache/clip_vlm_features',
        help="The directory containing the 'clip_vlm_features' cache to clean."
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help="Actually delete the files. If not set, performs a dry run."
    )
    args = parser.parse_args()

    scan_and_delete_7b_cache(Path(args.cache_dir), dry_run=not args.delete)

if __name__ == "__main__":
    main()
