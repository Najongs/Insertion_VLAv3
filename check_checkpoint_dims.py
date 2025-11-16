import torch
import sys

def check_checkpoint(path):
    """Loads a checkpoint and prints the shape of each tensor."""
    try:
        # Redirect output to a file to avoid cluttering the console
        original_stdout = sys.stdout
        with open('checkpoint_dims.txt', 'w') as f:
            sys.stdout = f

            print(f"Inspecting checkpoint: {path}")
            print("="*80)

            checkpoint = torch.load(path, map_location='cpu')
            
            state_dict = None
            # The actual model state_dict can be nested in different ways
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'sensor_encoder_state_dict' in checkpoint:
                state_dict = checkpoint['sensor_encoder_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                print("Could not find a state_dict in the checkpoint.", file=sys.stderr)
                return

            if not state_dict:
                print("State dictionary is empty or not found.", file=sys.stderr)
                return

            max_key_len = max(len(k) for k in state_dict.keys()) if state_dict else 0
            
            for key, value in state_dict.items():
                print(f"{key.ljust(max_key_len)} : {value.shape}")
                
            print("="*80)
            print("Inspection complete.")

    except Exception as e:
        print(f"Error loading or inspecting checkpoint: {e}", file=sys.stderr)
    finally:
        sys.stdout = original_stdout
        print("Checkpoint inspection is complete. The results have been saved to checkpoint_dims.txt.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        check_checkpoint(checkpoint_path)
    else:
        print("Please provide the path to the checkpoint file.", file=sys.stderr)
        print("Usage: python3 check_checkpoint_dims.py <path_to_checkpoint>", file=sys.stderr)
