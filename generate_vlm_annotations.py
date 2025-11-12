import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
from pathlib import Path
import re
import argparse

# ==============================
# Configuration
# ==============================
DEFAULT_DATASET_ROOTS = [
    "/home/najo/NAS/VLA/dataset/New_dataset",
    "/home/najo/NAS/VLA/dataset/New_dataset2"
]
VLM_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_FILE = "vlm_annotations.json"
HAND_EYE_VIEW_KEYWORD = "View5"


# ==============================
# ✅ Target detection rule
# ==============================
def is_target_found(response, frame_idx, task_name):
    text = response.lower()

    # ------------------------------------------
    # 1. Must include HIGH confidence
    # ------------------------------------------
    if ("confidence: high" not in text) and ("confidence high" not in text):
        return False

    # ------------------------------------------
    # 2. Must not say "target out of view"
    # ------------------------------------------
    if "target out of view" in text:
        return False

    # ------------------------------------------
    # 3. Check for READY_FOR_INSERTION (all tasks)
    # ------------------------------------------
    if "ready_for_insertion" in text:
        return True

    return False


# ==============================
# ✅ Task-specific VLM prompt
# ==============================
def get_prompt_for_task(task_name):

    # ----------- Eye Trocar task ----------------
    if "eye" in task_name.lower() or "trocar" in task_name.lower():
        return (
            "You are an intelligent robotic assistant observing a surgical scene through a hand-eye camera. "
            "Your current task is **trocar insertion into an eye phantom**. "
            "The environment includes an optical breadboard, but the pegboard is not your target. "
            "Focus on the **trocar**, which is a small cylindrical part with a central hole, and the **needle** attached to the robot arm. "
            "You need to identify the **target (trocar)** and assess its position relative to the needle. "
            "Consider how the robot should move to achieve insertion. "
            "Describe the current situation, including the surrounding objects and environment relevant to the task.\n\n"
            "You MUST output the following structure:\n"
            "1) Target visibility: FULLY_VISIBLE / PARTIALLY_VISIBLE / NOT_VISIBLE\n"
            "2) Needle distance: FAR / MID / NEAR\n"
            "3) Confidence: HIGH / MEDIUM / LOW\n"
            "4) If FULLY_VISIBLE and NEAR, optionally add 'READY_FOR_INSERTION'.\n"
            "If NOT_VISIBLE, say 'target out of view'."
        )

    # ----------- Colored-dot target task ----------
    else:
        # Extract color from task name (e.g., "Green_point" -> "green")
        color = "colored"
        task_lower = task_name.lower()
        if "green" in task_lower:
            color = "green"
        elif "yellow" in task_lower:
            color = "yellow"
        elif "blue" in task_lower:
            color = "blue"
        elif "red" in task_lower:
            color = "red"
        elif "white" in task_lower:
            color = "white"

        return (
            f"You are an intelligent robotic assistant observing a robotic needle insertion task through a hand-eye camera. "
            f"Your current task is **inserting a needle into a {color} dot target on a silicone surface**. "
            f"The environment includes an optical breadboard, but the pegboard is not your target. "
            f"Focus on the **needle** attached to the robot arm and the **{color} dot target**. "
            f"You need to identify the **target ({color} dot)** and assess its position relative to the needle. "
            f"Consider how the robot should move to achieve insertion. "
            f"Describe the current situation, including the surrounding objects and environment relevant to the task.\n\n"
            f"Output the following structure:\n"
            f"1) Target visibility: FULLY_VISIBLE / PARTIALLY_VISIBLE / NOT_VISIBLE\n"
            f"2) Needle distance: FAR / MID / NEAR / VERY_NEAR\n"
            f"3) Confidence: HIGH / MEDIUM / LOW\n"
            f"4) If VERY_NEAR and needle tip is almost touching the target, add 'READY_FOR_INSERTION'.\n"
            f"If NOT_VISIBLE, say 'target out of view'."
        )


# ==============================
# ✅ Generate VLM description
# ==============================
def generate_vlm_description(image_path, model, processor, task_name):
    image = Image.open(image_path).convert("RGB")
    prompt = get_prompt_for_task(task_name)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=800, do_sample=False, num_beams=1)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    return response


# ==============================
# ✅ MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Generate VLM annotations for robotic insertion tasks")
    parser.add_argument("--task_name", type=str, default=None,
                       help="Specific task to process (e.g., Eye_trocar, Green_point). If None, process all tasks.")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use (default: 0)")
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode: process only one episode")
    parser.add_argument("--dataset_roots", type=str, nargs='*', default=DEFAULT_DATASET_ROOTS,
                       help="Paths to dataset root directories (default: New_dataset and New_dataset2)")
    args = parser.parse_args()

    # Set GPU device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Starting VLM annotation generation on {device}...")

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    annotations = {}
    total_generated_count = 0

    # Process each dataset root
    for dataset_root in args.dataset_roots:
        if not os.path.exists(dataset_root):
            print(f"Warning: Dataset root not found: {dataset_root}, skipping...")
            continue

        print(f"\nProcessing dataset root: {dataset_root}")

        # Determine which tasks to process
        if args.task_name:
            task_folders = [args.task_name]
            print(f"Processing single task: {args.task_name}")
        else:
            task_folders = [
                d for d in os.listdir(dataset_root)
                if os.path.isdir(os.path.join(dataset_root, d))
            ]
            print(f"Processing all tasks: {task_folders}")

        for task_name in task_folders:
            # Skip if task folder doesn't exist
            task_path = os.path.join(dataset_root, task_name)
            if not os.path.exists(task_path):
                print(f"Warning: Task folder not found: {task_path}")
                continue

            tqdm.write(f"\nProcessing task: {task_name}")
            # Support both 'episode_*' and 'data_collection_*' folder patterns
            episode_folders = [
                e for e in os.listdir(task_path)
                if e.startswith("episode_") or e.startswith("data_collection_")
            ]

            for episode_id in tqdm(episode_folders, desc=f"Episodes in {task_name}"):
                episode_path = os.path.join(task_path, episode_id)
                annotations[episode_id] = {}
                target_found_timestamp = None

                try:
                    # Handle different folder structures
                    # Old format: episode_*/images/View5
                    # New format: data_collection_*/View5
                    if episode_id.startswith("episode_"):
                        image_dir = os.path.join(episode_path, "images", HAND_EYE_VIEW_KEYWORD)
                    else:  # data_collection_*
                        image_dir = os.path.join(episode_path, HAND_EYE_VIEW_KEYWORD)

                    image_files = sorted(
                        [f for f in os.listdir(image_dir) if f.endswith(".jpg")],
                        key=lambda f: re.search(r"(\d{10,}\.\d+)", f).group(1)
                    )

                    # Determine frame skip based on task
                    if "eye" in task_name.lower() or "trocar" in task_name.lower():
                        min_frame_skip = 100
                    else:
                        min_frame_skip = 200 # 60

                    for frame_idx, image_name in enumerate(image_files):
                        # Skip early frames
                        if frame_idx < min_frame_skip:
                            continue

                        timestamp_match = re.search(r"(\d{10,}\.\d+)\.jpg", image_name)
                        if not timestamp_match:
                            continue

                        timestamp = timestamp_match.group(1)
                        image_path = os.path.join(image_dir, image_name)

                        response = generate_vlm_description(image_path, model, processor, task_name)
                        annotations[episode_id][timestamp] = response
                        total_generated_count += 1

                        # ✅ Trigger detection
                        if target_found_timestamp is None and is_target_found(response, frame_idx, task_name):
                            target_found_timestamp = timestamp
                            tqdm.write(f"[TRIGGER] Target found at {timestamp} (frame {frame_idx})")
                            break

                except Exception as e:
                    tqdm.write(f"Warning: Failed to process {episode_id}. Error: {e}")

                annotations[episode_id]["target_found_timestamp"] = target_found_timestamp

                if args.test_mode:
                    print("Test mode: processed one episode, stopping.")
                    break

            if args.test_mode:
                break

        if args.test_mode:
            break

    # Save with task-specific filename if processing single task
    if args.task_name:
        output_filename = f"vlm_annotations_{args.task_name}.json"
    else:
        output_filename = OUTPUT_FILE

    output_path = Path.cwd() / output_filename
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=4)

    print("\n✅ Done.")
    print(f"Total annotations: {total_generated_count}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()