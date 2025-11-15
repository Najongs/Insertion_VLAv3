"""
Test CLIP VLM Cache Generation

Quick test to verify cache generation works correctly before full training.
Tests a few samples from each task and verifies the cache files are valid.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from pathlib import Path
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from TRAIN_SensorImage_CLIP import (
    CLIP_PROMPT_TEXT,
    get_clip_prompt_hash,
    get_formatted_clip_prompt,
    extract_task_name_from_episode_path,
    disable_generation_temperature,
    _generate_text_response_local,
)
from qwen_vl_utils import process_vision_info
from vla_cache_manager import VLACacheManager


def test_cache_generation():
    """Test cache generation with a few samples"""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache_root = Path("/home/najo/NAS/VLA/dataset/cache/clip_vlm_features")
    cache_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CLIP VLM Cache Generation Test")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Cache root: {cache_root}")

    # Load VLM model (use 7B to match training)
    print("\n⏳ Loading VLM model...")
    vlm_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"   Model: {vlm_model_name}")
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name, trust_remote_code=True)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2"
    )
    vlm_model.eval()
    disable_generation_temperature(vlm_model)
    print("✅ VLM model loaded")

    # Create cache manager
    cache_manager = VLACacheManager(cache_dir=str(cache_root))

    # Test samples: one episode from each task
    test_episodes = [
        ("/home/najo/NAS/VLA/dataset/New_dataset2/Red_point/data_collection_20251108_061254", "Red_point", 0),
        ("/home/najo/NAS/VLA/dataset/New_dataset2/White_point/data_collection_20251108_052043", "White_point", 0),
        ("/home/najo/NAS/VLA/dataset/New_dataset2/Green_point/data_collection_20251108_053719", "Green_point", 0),
    ]

    print(f"\n📋 Testing {len(test_episodes)} episodes...")

    results = []

    for episode_path, task_name, vlm_idx in tqdm(test_episodes, desc="Processing"):
        episode_path = Path(episode_path)
        episode_id = episode_path.name

        # Get prompt and hash for this task
        formatted_prompt = get_formatted_clip_prompt(task_name)
        task_prompt_hash = get_clip_prompt_hash(task_name)

        print(f"\n{'='*80}")
        print(f"Episode: {episode_id}")
        print(f"Task: {task_name}")
        print(f"Prompt hash: {task_prompt_hash}")

        # Find View5 image
        view5_dir = episode_path / "View5"
        if not view5_dir.exists():
            view5_dir = episode_path / "images" / "View5"

        if not view5_dir.exists():
            print(f"⚠️ View5 directory not found for {episode_id}")
            continue

        image_files = sorted(list(view5_dir.glob("*.jpg")) + list(view5_dir.glob("*.png")))
        if not image_files:
            print(f"⚠️ No images found in View5 for {episode_id}")
            continue

        # Load first image
        image = Image.open(image_files[vlm_idx]).convert("RGB")
        print(f"Image: {image_files[vlm_idx].name}, size: {image.size}")

        # Generate text response
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": formatted_prompt}
            ]
        }]
        generation_text_input = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vision_input, _ = process_vision_info(messages)

        text_response = _generate_text_response_local(
            vlm_model, vlm_processor, generation_text_input, vision_input, max_new_tokens=256
        )
        print(f"Text response: {text_response[:100]}...")

        with torch.no_grad():
            # 1. Image-only inference
            image_only_messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": ""}
                ]
            }]
            image_text_with_placeholders = vlm_processor.apply_chat_template(
                image_only_messages, tokenize=False, add_generation_prompt=False
            )
            image_only_vision_input, _ = process_vision_info(image_only_messages)
            image_inputs = vlm_processor(
                text=[image_text_with_placeholders],
                images=[image_only_vision_input],
                padding=True,
                return_tensors="pt"
            ).to(device=vlm_model.device, dtype=vlm_model.dtype)

            image_outputs = vlm_model(**image_inputs, output_hidden_states=True, return_dict=True)
            image_hidden_state = image_outputs.hidden_states[-1]

            # Extract image tokens (token ID 151655)
            image_token_mask = (image_inputs['input_ids'] == 151655)
            image_indices = torch.where(image_token_mask.squeeze(0))[0]
            image_features = image_hidden_state[:, image_indices, :]

            print(f"Image tokens found: {len(image_indices)}")
            print(f"Image features shape: {image_features.shape}")

            # 2. Text-only inference
            text_inputs = vlm_processor(
                text=[text_response],
                images=None,
                padding=True,
                return_tensors="pt"
            ).to(device=vlm_model.device, dtype=vlm_model.dtype)

            text_outputs = vlm_model(**text_inputs, output_hidden_states=True, return_dict=True)
            text_hidden_state = text_outputs.hidden_states[-1]
            guidance_vector = text_hidden_state.mean(dim=1)

            print(f"Guidance vector shape: {guidance_vector.shape}")

        # Save cache
        features_to_cache = (
            image_features.detach().to("cpu", dtype=torch.float16),
            guidance_vector.detach().to("cpu", dtype=torch.float16)
        )

        cache_manager.save_cache_tuple(
            dataset_name=episode_id,
            vlm_idx=vlm_idx,
            prompt_hash=task_prompt_hash,
            features_tuple=features_to_cache
        )

        # Verify cache
        cache_path = cache_root / task_prompt_hash / f"{episode_id}_vlm{vlm_idx}.pt"
        if cache_path.exists():
            loaded = torch.load(cache_path, map_location="cpu")
            if isinstance(loaded, tuple) and len(loaded) == 2:
                img_feat, guid_vec = loaded
                print(f"✅ Cache saved and verified:")
                print(f"   Path: {cache_path}")
                print(f"   Image features: {img_feat.shape}, dtype: {img_feat.dtype}")
                print(f"   Guidance vector: {guid_vec.shape}, dtype: {guid_vec.dtype}")

                # Check if image features are non-empty
                if img_feat.shape[1] > 0:
                    print(f"   ✅ Image features are non-empty ({img_feat.shape[1]} tokens)")
                    results.append({
                        "task": task_name,
                        "episode": episode_id,
                        "hash": task_prompt_hash,
                        "num_tokens": img_feat.shape[1],
                        "success": True
                    })
                else:
                    print(f"   ❌ WARNING: Image features are EMPTY!")
                    results.append({
                        "task": task_name,
                        "episode": episode_id,
                        "hash": task_prompt_hash,
                        "num_tokens": 0,
                        "success": False
                    })
            else:
                print(f"❌ Cache format error: {type(loaded)}")
                results.append({
                    "task": task_name,
                    "episode": episode_id,
                    "hash": task_prompt_hash,
                    "num_tokens": 0,
                    "success": False
                })
        else:
            print(f"❌ Cache file not found: {cache_path}")
            results.append({
                "task": task_name,
                "episode": episode_id,
                "hash": task_prompt_hash,
                "num_tokens": 0,
                "success": False
            })

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\nTotal tests: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    print("\nDetailed results:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{status} {r['task']:15s} | {r['hash']} | {r['num_tokens']:3d} tokens | {r['episode']}")

    if failed == 0:
        print("\n🎉 All tests passed! Cache generation is working correctly.")
        print("   You can now proceed with full training.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

    # Cleanup
    del vlm_model
    del vlm_processor
    torch.cuda.empty_cache()

    return failed == 0


if __name__ == "__main__":
    success = test_cache_generation()
    exit(0 if success else 1)
