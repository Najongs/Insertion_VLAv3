"""
Inspect Cache Files

A general-purpose script to inspect the contents of .pt cache files.
It loads a given .pt file and prints a recursive summary of its structure,
including types, lengths, shapes, and dtypes.
"""
import argparse
from pathlib import Path
import torch

def inspect_structure(data, indent=0):
    """Recursively prints the structure of the loaded data."""
    prefix = "  " * indent
    if isinstance(data, torch.Tensor):
        print(f"{prefix}Tensor(shape={data.shape}, dtype={data.dtype})")
    elif isinstance(data, tuple):
        print(f"{prefix}Tuple(len={len(data)})")
        for i, item in enumerate(data):
            print(f"{prefix}- Item {i}:")
            inspect_structure(item, indent + 1)
    elif isinstance(data, list):
        print(f"{prefix}List(len={len(data)})")
        # Only inspect the first element of a list to avoid spam
        if data:
            print(f"{prefix}- Item 0 (example):")
            inspect_structure(data[0], indent + 1)
    elif isinstance(data, dict):
        print(f"{prefix}Dict(keys={list(data.keys())})")
        for key, value in data.items():
            print(f"{prefix}- Key '{key}':")
            inspect_structure(value, indent + 1)
    else:
        print(f"{prefix}{type(data).__name__}")

def main():
    parser = argparse.ArgumentParser(description="Inspect the contents of a .pt cache file.")
    parser.add_argument(
        'file_paths', 
        type=str, 
        nargs='+',
        help="Path(s) to the .pt file(s) to inspect."
    )
    args = parser.parse_args()

    for file_path_str in args.file_paths:
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"\n--- File Not Found: {file_path} ---")
            continue

        print(f"\n--- Inspecting: {file_path} ---")
        try:
            content = torch.load(file_path, map_location='cpu')
            inspect_structure(content)
        except Exception as e:
            print(f"Error loading or inspecting file: {e}")
        print("-" * (20 + len(str(file_path))))

if __name__ == "__main__":
    main()
