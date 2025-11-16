import torch
import time
import numpy as np
from models.unified_model import QwenVLAUnified
from PIL import Image
import os
import shutil

def create_dummy_image(width=640, height=360):
    """Creates a dummy PIL image."""
    return Image.fromarray(np.uint8(np.random.rand(height, width, 3) * 255))

def run_inference_test():
    """
    Tests the inference time of the QwenVLAUnified model with an increasing number of image views.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU. Timing results may not be representative.")

    print("Initializing QwenVLAUnified model for parallel processing test...")
    # Initialize the model with parallel view encoding and pooling (weighted_mean).
    model = QwenVLAUnified(
        model_type='flow_matching',
        sensor_enabled=True,
        robot_state_enabled=False,
        finetune_vl='none',
        cache_dir="./test_cache_inference",
        image_resize_height=360,
        image_resize_width=640,
        parallel_view_encoding=True,
        view_aggregation='mean'
    ).to(device)
    model.eval()
    print("Model initialized.")

    batch_size = 1
    text_inputs_dummy = ["Move the robot arm to grasp the cup."]
    # Create dummy sensor data based on default model config (65, 1026)
    sensor_data_dummy = torch.randn(batch_size, 65, 1026, device=device, dtype=torch.bfloat16)

    # --- Test with increasing number of views ---
    max_views = 5
    iterations = 10
    print(f"\nStarting inference time test for 1 to {max_views} views...")
    print(f"Running {iterations} iterations per view count.")

    for num_views in range(1, max_views + 1):
        # The model's processor expects a list of PIL Images.
        dummy_images = [create_dummy_image(640, 360) for _ in range(num_views)]
        image_inputs_dummy = [dummy_images] * batch_size # [[img1, img2, ...]] for batch size 1

        # Warm-up run to handle any initial CUDA overhead
        print(f"\n--- Warming up for {num_views} view(s)... ---")
        with torch.no_grad():
            _ = model.predict_action(
                text_inputs=text_inputs_dummy,
                image_inputs=image_inputs_dummy,
                sensor_data=sensor_data_dummy,
                cache=False # Disable caching for timing test
            )
        
        if device == 'cuda':
            torch.cuda.synchronize()

        # Timing loop
        timings = []
        print(f"--- Running timed inference for {num_views} view(s)... ---")
        for i in range(iterations):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model.predict_action(
                    text_inputs=text_inputs_dummy,
                    image_inputs=image_inputs_dummy,
                    sensor_data=sensor_data_dummy,
                    cache=False # Disable caching for timing test
                )
            if device == 'cuda':
                torch.cuda.synchronize() # Wait for the GPU to finish
            end_time = time.perf_counter()
            timings.append(end_time - start_time)

        avg_time = np.mean(timings)
        std_dev = np.std(timings)
        
        print(f"▶ Results for {num_views} view(s):")
        print(f"  Average inference time: {avg_time:.4f} seconds")
        print(f"  Standard deviation:     {std_dev:.4f} seconds")
        # print(f"  Individual timings: {[float(f'{t:.4f}') for t in timings]}")


if __name__ == "__main__":
    try:
        import PIL
    except ImportError:
        print("Pillow is not installed. Installing...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install Pillow: {e}")
            print("Please install it manually: pip install Pillow")
            sys.exit(1)

    try:
        run_inference_test()
    finally:
        # Clean up the cache directory created during the test
        test_cache_dir = "./test_cache_inference"
        if os.path.exists(test_cache_dir):
            print(f"\nCleaning up test cache directory: {test_cache_dir}")
            shutil.rmtree(test_cache_dir)
            print("Cleanup complete.")
