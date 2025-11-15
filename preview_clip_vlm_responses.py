"""
Preview CLIP VLM Text Responses

Qwen2.5-VL 모델이 CLIP 학습을 위해 어떤 텍스트 응답을 생성하는지 확인하는 스크립트.
- View5와 View4 (hand_eye) 이미지 사용
- 에피소드의 마지막 20% 샘플만 처리 (80% 이후)
- CLIP_PROMPT_TEXT 사용

Usage:
    python preview_clip_vlm_responses.py \
        --episode_dir /path/to/episode \
        --output_dir ./clip_vlm_preview \
        --num_samples 10 \
        --vlm_model Qwen/Qwen2.5-VL-3B-Instruct \
        --hand_eye_views View5 View4
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Import CLIP prompt from training script
from TRAIN_SensorImage_CLIP import (
    CLIP_PROMPT_TEXT,
    get_clip_prompt_hash,
    get_formatted_clip_prompt,
    extract_task_name_from_episode_path
)


def generate_vlm_response(
    model,
    processor,
    images,
    instruction: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
):
    """
    VLM 모델로부터 텍스트 응답을 생성합니다.

    Args:
        model: Qwen VL model
        processor: Qwen processor
        images: Single image or list with single image (View5 only)
        instruction: Text instruction/prompt (CLIP_PROMPT_TEXT)
        device: Device to run on
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        generated_text: VLM's text response
    """
    # Use only first image (View5)
    if isinstance(images, list):
        image = images[0] if len(images) > 0 else images
    else:
        image = images

    # Build content with single image
    content = [
        {"type": "image", "image": image},
        {"type": "text", "text": instruction}
    ]

    messages = [{"role": "user", "content": content}]

    # Process inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    vision_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=[vision_inputs] if vision_inputs else None,
        padding=True,
        return_tensors="pt"
    ).to(device, dtype=torch.bfloat16)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for consistency
            temperature=1.0,
        )

    # Decode (remove input prompt from output)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return generated_text


def preview_episode_clip_responses(
    episode_dir: str,
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    num_samples: int = 10,
    output_dir: str = "./clip_vlm_preview",
    device: str = "cuda",
    vlm_reuse_count: int = 1,
    hand_eye_view_keywords: list = None,
):
    """
    한 에피소드에서 CLIP VLM 응답을 미리보기합니다.
    - 마지막 20% 샘플만 처리 (80% 이후)
    - View5와 View4 (hand_eye) 이미지 사용

    Args:
        episode_dir: Episode directory path
        vlm_model_name: VLM model name
        num_samples: Number of samples to preview (from last 20%)
        output_dir: Output directory for saving responses
        device: Device to run on
        vlm_reuse_count: VLM reuse count (usually 1 for CLIP cache building)
        hand_eye_view_keywords: List of keywords to identify hand-eye views (default: ["View5"])
    """
    if hand_eye_view_keywords is None:
        hand_eye_view_keywords = ["View5"]
    episode_path = Path(episode_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Episode: {episode_path.name}")
    print(f"🤖 VLM Model: {vlm_model_name}")
    print(f"📊 Samples to preview: {num_samples} (from last 20%)")
    print(f"💾 Output directory: {output_path}")
    print(f"🎯 Hand-eye views: {', '.join(hand_eye_view_keywords)}")
    print()

    # Load model and processor
    print("⏳ Loading VLM model...")
    processor = AutoProcessor.from_pretrained(vlm_model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("✅ Model loaded")
    print()

    # Extract task name from episode path
    # Example: /dataset/New_dataset2/Red_point/data_collection_xxx → "Red_point"
    task_name = extract_task_name_from_episode_path(episode_path)
    print(f"🎯 Task name: {task_name}")

    # Use CLIP prompt with task_name filled in
    instruction = get_formatted_clip_prompt(task_name)
    prompt_hash = get_clip_prompt_hash(task_name)
    print(f"📝 CLIP Prompt Hash: {prompt_hash}")
    print(f"📝 Instruction:\n{instruction}")
    print()

    # Find all hand-eye view directories
    hand_eye_dirs = []
    for view_dir in episode_path.iterdir():
        if view_dir.is_dir():
            for keyword in hand_eye_view_keywords:
                if keyword.lower() in view_dir.name.lower():
                    hand_eye_dirs.append(view_dir)
                    break

    if not hand_eye_dirs:
        raise ValueError(f"No hand-eye view directories found matching {hand_eye_view_keywords} in {episode_path}")

    # Collect images per view directory (to maintain sync between views)
    view_images_dict = {}
    for hand_eye_dir in hand_eye_dirs:
        images = sorted(list(hand_eye_dir.glob("*.png")) + list(hand_eye_dir.glob("*.jpg")))
        if images:
            view_images_dict[hand_eye_dir.name] = images
            print(f"🖼️  Found hand-eye view: {hand_eye_dir.name}")
            print(f"   Total images: {len(images)}")

    if not view_images_dict:
        raise ValueError(f"No images found in any hand-eye view directory")

    # Use the first view to determine the number of timesteps
    first_view_name = list(view_images_dict.keys())[0]
    total_images = len(view_images_dict[first_view_name])
    print(f"\n📊 Using {first_view_name} as reference for timesteps: {total_images} images")
    print()

    # Strategy: Use last 20% of episode data (80% onwards)
    start_idx = int(total_images * 0.2)
    candidate_indices = list(range(start_idx, total_images))

    print(f"📊 Episode statistics:")
    print(f"   Total images: {total_images}")
    print(f"   80% threshold: {start_idx}")
    print(f"   Last 20% images: {len(candidate_indices)} samples")
    print()

    # Select samples to preview
    samples_to_preview = min(num_samples, len(candidate_indices))
    # Evenly sample from the last 20%
    if samples_to_preview < len(candidate_indices):
        step = len(candidate_indices) / samples_to_preview
        sample_indices = [int(i * step) for i in range(samples_to_preview)]
    else:
        sample_indices = list(range(len(candidate_indices)))

    print(f"🎯 Will preview {samples_to_preview} samples")
    print(f"   Sample indices (within last 20%): {sample_indices}")
    print()

    # Generate responses
    results = []
    print("🚀 Generating CLIP VLM responses...")
    print()

    # Initialize JSON file
    output_file = output_path / f"{episode_path.name}_clip_vlm_responses.json"
    initial_data = {
        "episode_name": episode_path.name,
        "episode_path": str(episode_path),
        "task_name": task_name,
        "vlm_model": vlm_model_name,
        "instruction": instruction,
        "prompt_hash": prompt_hash,
        "hand_eye_views": list(view_images_dict.keys()),
        "total_episode_images": total_images,
        "start_idx_80_percent": start_idx,
        "last_20_percent_images": len(candidate_indices),
        "samples_previewed": 0,
        "results": [],
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, indent=2, ensure_ascii=False)

    for local_idx in tqdm(sample_indices, desc="Processing"):
        # local_idx is index within candidate_indices (0-based for last 20%)
        # candidate_indices already contains global indices (start_idx to total_images)
        global_idx = candidate_indices[local_idx]

        # Calculate VLM index (matching dataset logic)
        vlm_idx = (global_idx // vlm_reuse_count) * vlm_reuse_count

        # Collect images from all views for this timestep
        timestep_images = []
        view_names = []
        image_paths_str = []

        for view_name, images_list in view_images_dict.items():
            if global_idx < len(images_list):
                image_path = images_list[global_idx]
                timestep_images.append(str(image_path))
                view_names.append(view_name)
                image_paths_str.append(str(image_path))

        if not timestep_images:
            print(f"⚠️ Warning: No images found for global_idx {global_idx}")
            continue

        # Generate response with multiple images
        try:
            generated_text = generate_vlm_response(
                model=model,
                processor=processor,
                images=timestep_images,  # Pass list of images
                instruction=instruction,
                device=device,
                max_new_tokens=256,
            )

            result = {
                "global_sample_idx": int(global_idx),
                "local_idx_in_last_20_percent": int(local_idx),
                "vlm_idx": int(vlm_idx),
                "view_names": view_names,  # Changed to list
                "image_paths": image_paths_str,  # Changed to list
                "instruction": instruction,
                "generated_response": generated_text,
            }
            results.append(result)

            # Append result to JSON file in real-time
            with open(output_file, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data['results'].append(result)
                data['samples_previewed'] = len(data['results'])
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.truncate()

            # Print preview
            print(f"\n{'='*80}")
            print(f"Global Sample #{global_idx} (Last 20% local idx: {local_idx}, VLM idx: {vlm_idx})")
            print(f"{'='*80}")
            for view_name, img_path in zip(view_names, image_paths_str):
                img_path_obj = Path(img_path)
                print(f"🖼️  View: {view_name}")
                print(f"    Image: {img_path_obj.parent.name}/{img_path_obj.name}")
            print(f"\n🤖 VLM Response:")
            print(f"   {generated_text}")
            print()

        except Exception as e:
            print(f"⚠️ Error processing sample {global_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print()
    print(f"✅ Results saved to: {output_file}")
    print(f"📊 Generated {len(results)} CLIP VLM responses")

    # Create a readable text summary
    summary_file = output_path / f"{episode_path.name}_clip_vlm_responses.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"CLIP VLM Response Preview\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Episode: {episode_path.name}\n")
        f.write(f"Task Name: {task_name}\n")
        f.write(f"VLM Model: {vlm_model_name}\n")
        f.write(f"Prompt Hash: {prompt_hash}\n")
        f.write(f"Hand-eye Views: {', '.join(view_images_dict.keys())}\n")
        f.write(f"Total Episode Images: {total_images}\n")
        f.write(f"80% Threshold Index: {start_idx}\n")
        f.write(f"Last 20% Images: {len(candidate_indices)}\n")
        f.write(f"Samples Previewed: {len(results)}\n\n")
        f.write(f"Instruction:\n{instruction}\n\n")

        for result in results:
            f.write(f"{'='*80}\n")
            f.write(f"Global Sample #{result['global_sample_idx']} ")
            f.write(f"(Last 20% idx: {result['local_idx_in_last_20_percent']}, ")
            f.write(f"VLM idx: {result['vlm_idx']})\n")
            f.write(f"{'='*80}\n\n")

            # Handle both old (single view) and new (multiple views) formats
            if 'view_names' in result and 'image_paths' in result:
                for view_name, img_path in zip(result['view_names'], result['image_paths']):
                    f.write(f"View: {view_name}\n")
                    f.write(f"Image: {Path(img_path).parent.name}/{Path(img_path).name}\n")
            else:
                # Legacy format support
                f.write(f"View: {result.get('view_name', 'N/A')}\n")
                f.write(f"Image: {Path(result.get('image_path', '')).parent.name}/{Path(result.get('image_path', '')).name}\n")

            f.write(f"\nVLM Response:\n")
            f.write(f"{result['generated_response']}\n\n")

    print(f"📄 Text summary saved to: {summary_file}")
    print()
    print("🎉 Done!")


def main():
    parser = argparse.ArgumentParser(description='Preview CLIP VLM text responses for an episode')

    parser.add_argument('--episode_dir', type=str, required=True,
                        help='Path to episode directory')
    parser.add_argument('--vlm_model', type=str,
                        default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='VLM model name (default: Qwen2.5-VL-3B-Instruct)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to preview from last 20%% (default: 10)')
    parser.add_argument('--output_dir', type=str, default='./clip_vlm_preview',
                        help='Output directory for saving responses')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (default: cuda)')
    parser.add_argument('--vlm_reuse_count', type=int, default=1,
                        help='VLM reuse count (default: 1 for CLIP cache building)')
    parser.add_argument('--hand_eye_views', type=str, nargs='+', default=['View5'],
                        help='Hand-eye view keywords (default: View5)')

    args = parser.parse_args()

    preview_episode_clip_responses(
        episode_dir=args.episode_dir,
        vlm_model_name=args.vlm_model,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device,
        vlm_reuse_count=args.vlm_reuse_count,
        hand_eye_view_keywords=args.hand_eye_views,
    )


if __name__ == "__main__":
    main()
