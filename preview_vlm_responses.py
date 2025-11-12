"""
Preview VLM Text Responses

Qwen2.5-VL 모델이 실제로 어떤 텍스트 응답을 생성하는지 확인하는 스크립트.
한 에피소드의 샘플들에 대해 VLM의 전체 생성 결과를 저장합니다.

Usage:
    python preview_vlm_responses.py \
        --episode_dir /path/to/episode \
        --output_dir ./vlm_preview \
        --num_samples 10 \
        --vlm_model Qwen/Qwen2.5-VL-3B-Instruct
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def generate_vlm_response(
    model,
    processor,
    images: list,
    instruction: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
):
    """
    VLM 모델로부터 텍스트 응답을 생성합니다.

    Args:
        model: Qwen VL model
        processor: Qwen processor
        images: List of image paths or PIL Images
        instruction: Text instruction/prompt
        device: Device to run on
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        generated_text: VLM's text response
    """
    # Prepare messages
    content = []
    for img in images:
        if isinstance(img, str):
            content.append({"type": "image", "image": img})
        else:
            content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": instruction})

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


def preview_episode_responses(
    episode_dir: str,
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    num_samples: int = 10,
    output_dir: str = "./vlm_preview",
    device: str = "cuda",
    vlm_reuse_count: int = 3,
):
    """
    한 에피소드에서 VLM 응답을 미리보기합니다.

    Args:
        episode_dir: Episode directory path
        vlm_model_name: VLM model name
        num_samples: Number of samples to preview
        output_dir: Output directory for saving responses
        device: Device to run on
        vlm_reuse_count: VLM reuse count (same as dataset config)
    """
    episode_path = Path(episode_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Episode: {episode_path.name}")
    print(f"🤖 VLM Model: {vlm_model_name}")
    print(f"📊 Samples to preview: {num_samples}")
    print(f"💾 Output directory: {output_path}")
    print()

    # Load model and processor
    print("⏳ Loading VLM model...")
    processor = AutoProcessor.from_pretrained(vlm_model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print("✅ Model loaded")
    print()

    # Load metadata and construct instruction
    metadata_path = episode_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    task_name = episode_path.parent.name.replace("_", " ")
    instruction = f"""Respond ONLY with the next action.
Environment Context:
- This is a Meca500 robot workspace.
- The end-effector holds a needle; the needle tip is the tool.
- The scene is an optical table with many holes, but these are NOT targets.
- The ONLY true insertion target is the {task_name}.

Task:
You must analyze the five camera views and determine the needle’s relative position to the {task_name}.
Identify:
1) needle tip location
2) alignment relative to the {task_name} center
3) required direction to align for insertion

Respond with:
- target visibility
- needle alignment
- required adjustment direction
- insertion readiness (yes/no)
"""
    print(f"📝 Instruction:\n{instruction}")
    print()

    # Get image directories (new format)
    image_dirs = {}
    for view_dir in episode_path.iterdir():
        if view_dir.is_dir() and view_dir.name.startswith("View"):
            view_name = view_dir.name
            # Look for both png and jpg files
            images = sorted(list(view_dir.glob("*.png")) + list(view_dir.glob("*.jpg")))
            if images:
                image_dirs[view_name] = [str(f) for f in images]

    if not image_dirs:
        raise ValueError(f"No image directories found in {episode_path}")

    print(f"🖼️  Found {len(image_dirs)} views:")
    for view_name, images in image_dirs.items():
        print(f"   - {view_name}: {len(images)} images")
    print()

    # Determine total samples
    max_images = max(len(imgs) for imgs in image_dirs.values())
    total_samples = min(num_samples, max_images)

    # Calculate VLM indices (matching dataset logic)
    vlm_indices = []
    for sample_idx in range(total_samples):
        vlm_idx = (sample_idx // vlm_reuse_count) * vlm_reuse_count
        vlm_indices.append((sample_idx, vlm_idx))

    # Remove duplicates (only keep first occurrence of each vlm_idx)
    unique_vlm_indices = {}
    for sample_idx, vlm_idx in vlm_indices:
        if vlm_idx not in unique_vlm_indices:
            unique_vlm_indices[vlm_idx] = sample_idx

    print(f"🎯 VLM update points: {list(unique_vlm_indices.keys())}")
    print(f"   (VLM reuse count: {vlm_reuse_count})")
    print()

    # Generate responses
    results = []
    print("🚀 Generating VLM responses...")
    print()

    # Initialize JSON file
    output_file = output_path / f"{episode_path.name}_vlm_responses.json"
    initial_data = {
        "episode_name": episode_path.name,
        "episode_path": str(episode_path),
        "vlm_model": vlm_model_name,
        "instruction": instruction,
        "vlm_reuse_count": vlm_reuse_count,
        "total_samples": total_samples,
        "num_vlm_updates": 0,
        "results": [],
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, indent=2, ensure_ascii=False)

    for vlm_idx, sample_idx in tqdm(sorted(unique_vlm_indices.items()), desc="Processing"):
        # Collect images from all views at this VLM index
        image_paths = []
        for view_name in sorted(image_dirs.keys()):
            images = image_dirs[view_name]
            img_idx = min(vlm_idx, len(images) - 1)
            image_paths.append(str(images[img_idx]))

        # Generate response
        try:
            generated_text = generate_vlm_response(
                model=model,
                processor=processor,
                images=image_paths,
                instruction=instruction,
                device=device,
                max_new_tokens=256,
            )

            result = {
                "sample_idx": int(sample_idx),
                "vlm_idx": int(vlm_idx),
                "instruction": instruction,
                "image_paths": image_paths,
                "generated_response": generated_text,
            }
            results.append(result)

            # Append result to JSON file in real-time
            with open(output_file, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data['results'].append(result)
                data['num_vlm_updates'] = len(data['results'])
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.truncate()

            # Print preview
            print(f"\n{'='*80}")
            print(f"Sample #{sample_idx} (VLM update at idx {vlm_idx})")
            print(f"{'='*80}")
            print(f"📝 Instruction: {instruction}")
            print(f"🖼️  Images: {len(image_paths)} views")
            for i, img_path in enumerate(image_paths, 1):
                print(f"   {i}. {Path(img_path).parent.name}/{Path(img_path).name}")
            print(f"\n🤖 VLM Response:")
            print(f"   {generated_text}")
            print()

        except Exception as e:
            print(f"⚠️ Error processing sample {sample_idx}: {e}")
            continue

    print()
    print(f"✅ Results saved to: {output_file}")
    print(f"📊 Generated {len(results)} VLM responses for {total_samples} samples")

    # Create a readable text summary
    summary_file = output_path / f"{episode_path.name}_vlm_responses.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"VLM Response Preview\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Episode: {episode_path.name}\n")
        f.write(f"VLM Model: {vlm_model_name}\n")
        f.write(f"Instruction: {instruction}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"VLM Reuse Count: {vlm_reuse_count}\n")
        f.write(f"Number of VLM Updates: {len(results)}\n\n")

        for result in results:
            f.write(f"{'='*80}\n")
            f.write(f"Sample #{result['sample_idx']} (VLM update at idx {result['vlm_idx']})\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Images ({len(result['image_paths'])} views):\n")
            for i, img_path in enumerate(result['image_paths'], 1):
                f.write(f"  {i}. {Path(img_path).parent.name}/{Path(img_path).name}\n")
            f.write(f"\nVLM Response:\n")
            f.write(f"{result['generated_response']}\n\n")

    print(f"📄 Text summary saved to: {summary_file}")
    print()
    print("🎉 Done!")


def main():
    parser = argparse.ArgumentParser(description='Preview VLM text responses for an episode')

    parser.add_argument('--episode_dir', type=str, required=True,
                        help='Path to episode directory')
    parser.add_argument('--vlm_model', type=str,
                        default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='VLM model name (default: Qwen2.5-VL-3B-Instruct)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to preview (default: 10)')
    parser.add_argument('--output_dir', type=str, default='./vlm_preview',
                        help='Output directory for saving responses')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (default: cuda)')
    parser.add_argument('--vlm_reuse_count', type=int, default=3,
                        help='VLM reuse count (default: 3, same as training)')

    args = parser.parse_args()

    preview_episode_responses(
        episode_dir=args.episode_dir,
        vlm_model_name=args.vlm_model,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device,
        vlm_reuse_count=args.vlm_reuse_count,
    )


if __name__ == "__main__":
    main()
