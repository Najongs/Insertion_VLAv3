"""
Sensor Encoder Pre-training Script using CLIP-style Contrastive Learning

This script pre-trains the SensorEncoder by matching its output with the
features from a Vision-Language Model (VLM) that sees the corresponding
hand-eye camera view.

Key implementation notes for quick reference:
- ForceAwareSensorEncoder splits the 1026-channel sensor input into distance (1025 ch) and force (1 ch). The distance stream uses the ConvFormer-style SensorEncoder while the force stream uses a small MLP + temporal pooling; their outputs are concatenated (∼896 dims from distance, 128 dims from force when `sensor_output_dim=1024`).
- CLIPModel loads cached Qwen2.5-VL features where each sample stores `vlm_image_features` (token embeddings, dim=2048) and a `vlm_guidance_vector` (text response mean, dim=2048). The guidance vector attends over the image tokens via `nn.MultiheadAttention` before being projected.
- Both sensor and fused VLM embeddings are projected down to the shared `embedding_dim` (default 512) via linear heads + LayerNorm. Only this projected space participates in the SigLIP contrastive loss; the underlying 1024-d sensor features remain unchanged for downstream use.
- Parameters updated during training include the entire sensor encoder, the VLM fusion attention module, and the projection heads. Cached VLM features stay frozen.

⚠️ CLIP VLM 캐시 구조 및 prompt_hash 매칭 (중요!):
------------------------------------------------------------------------------
이 스크립트는 CLIP 학습을 위한 별도의 VLM 캐시를 사용합니다:
  - 캐시 경로: {cache_root}/clip_vlm_features/{prompt_hash}/{episode_name}_vlm{idx}.pt
  - prompt_hash: CLIP_PROMPT_TEXT (line 59-67)를 MD5 해시화한 값
  - 주의: Flow Matching의 VL 캐시와는 다른 prompt와 경로 사용!

Flow Matching VL 캐시와의 차이점:
  1. Flow Matching: task_name이 포함된 instruction 사용
     - 예: "...target is the Red point..." → prompt_hash는 태스크별로 다름
     - 경로: {cache_root}/qwen_vl_features/{task_specific_hash}/

  2. CLIP VLM: 고정된 CLIP_PROMPT_TEXT 사용
     - 모든 태스크에 동일한 prompt 사용 (task_name 없음)
     - 경로: {cache_root}/clip_vlm_features/{clip_hash}/
     - prompt_hash는 CLIP_PROMPT_TEXT에만 의존 (태스크 무관)

캐시 생성 및 매칭:
  - 캐시는 build_clip_cache_for_dataset() 함수로 수동/선행 생성 가능
  - prompt_hash는 get_clip_prompt_hash() 함수로 계산 (line 90대)
  - CLIP_PROMPT_TEXT를 변경하면 prompt_hash도 변경되어 기존 캐시를 못 찾음!
  - 학습 시에는 VLM을 GPU로 로드하지 않으며, 캐시가 없는 샘플은 건너뜀(SKIP)

중요 참고사항:
  - CLIP 캐시는 에피소드의 마지막 20%만 사용 (접촉 이벤트 집중)
  - Flow Matching 캐시는 전체 100% 사용
  - 두 캐시는 독립적으로 관리됨

Methodology:
1. A batch consists of (sensor_data, hand_eye_image) pairs.
2. Sensor data is encoded by the SensorEncoder.
3. The hand-eye image is encoded by a VLM (e.g., Qwen2.5-VL) with a
   specific prompt asking it to identify contact events.
4. Both outputs are projected into a shared embedding space.
5. A contrastive loss (CLIP loss) pulls the embeddings of matching
   (sensor, image) pairs together and pushes non-matching pairs apart.
6. The trained SensorEncoder weights are saved for downstream tasks.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import re # Added for timestamp extraction
import hashlib
from typing import Literal, Tuple, List, Optional, Dict
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from pathlib import Path
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_model import ForceAwareSensorEncoder, force_bn_fp32_
from vla_datasets.unified_dataset import UnifiedVLADataset, create_unified_dataloader
from vla_cache_manager import VLACacheManager
from qwen_vl_utils import process_vision_info


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """Prints a summary of the model's parameters and estimated size."""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    if not is_main_process:
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimated size in MB (assuming float32, 4 bytes per parameter)
    total_size_mb = total_params * 4 / (1024 ** 2)
    trainable_size_mb = trainable_params * 4 / (1024 ** 2)
    
    print("\n" + "="*50)
    print(f"MODEL SUMMARY: {model_name}")
    print("="*50)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Non-Trainable Parameters: {total_params - trainable_params:,}")
    print("-" * 50)
    print(f"  Estimated Total Size: {total_size_mb:.2f} MB")
    print(f"  Estimated Trainable Size: {trainable_size_mb:.2f} MB")
    print("="*50 + "\n")


CLIP_PROMPT_TEXT = (
    "This view is from the robot's end-tool camera, focusing on the needle attached to a 3D-printed tool. "
    "I need to confirm what object the needle tip is approaching and whether it has been inserted. "
    "Please provide a concise explanation of the needle's current situation and, given it's in an approaching state, "
    "describe how close the needle tip might be to the target. The setup is on an optical table, so threaded holes are visible, "
    "but it is not piercing these; they are the floor surface far away from the needle. "
    "The target task is an insertion, and the target point is {task_name}. "
    "The fact that the target is not visible implies the robot is currently moving toward it. "
    "The insertion isn't always in the exact center of the target."
)
# 현재 사진에 대해 묘사하고, 로봇의 끝에 달린 3D 프린팅 Tool에 바늘에 집중해서 바늘 끝이 어떤 물체와 근접하고 삽입이 된건지 확인이 필요해. 간결하게 답변하며, 바늘의 현재 상황과 바늘 끝단이 접근하고 있는 상태라 바늘끝이 앞의 대상과 얼마나 가까울지 설명해줘 지금 뷰는 로봇 End tool에 달린 카메라의 뷰야, 현재 상황은 광학 테이블 위에 존재하고 있어서 나사 구멍이 보일텐데 여길 찌르고 있는게 아니라 바늘과 거리가 멀리 떨어진 바닥면이야. 목표하는 태스크가 어딘가에 찌르는 건데 목표 지점은 Red_point 야, 목표지점이 안보인다는건 그쪽으로 이동중이란 거야. 꼭 타겟 가운데에서 바늘을 찌르는 건 아냐. 
SIGLIP_TEMPERATURE = 0.07

def disable_generation_temperature(vlm_model):
    """
    Neutralize the generation temperature without removing the attribute
    so downstream code that expects it still works.
    """
    gen_cfg = getattr(vlm_model, "generation_config", None)
    if gen_cfg is None:
        return

    # Drop from the "extra" dict that triggers HF warnings
    extra_params = getattr(gen_cfg, "_extra_generation_params", None)
    if isinstance(extra_params, dict):
        extra_params.pop("temperature", None)

    # Keep the attribute present but inactive
    try:
        setattr(gen_cfg, "temperature", None)
    except AttributeError:
        gen_cfg.__dict__["temperature"] = None


def extract_task_name_from_episode_path(episode_path: Path) -> str:
    """
    Extract task name from episode directory path.
    Example: /dataset/New_dataset2/Red_point/data_collection_xxx → "Red_point"
    """
    return episode_path.parent.name


def get_formatted_clip_prompt(task_name: str) -> str:
    """Get CLIP prompt with task_name filled in."""
    return CLIP_PROMPT_TEXT.format(task_name=task_name)


def get_clip_prompt_hash(task_name: str = "Unknown") -> str:
    """
    Get prompt hash for given task name.
    Each task (Red_point, White_point, etc.) will have a different hash.
    """
    prompt = CLIP_PROMPT_TEXT.format(task_name=task_name)
    return hashlib.md5(prompt.encode()).hexdigest()[:8]


def _generate_text_response_local(
    vlm_model,
    vlm_processor,
    generation_text_input,
    vision_input,
    max_new_tokens,
):
    """Generate the VLM's textual response for the given prompt/image pair."""
    model_inputs = vlm_processor(
        text=[generation_text_input],
        images=[vision_input],
        padding=True,
        return_tensors="pt",
    ).to(device=vlm_model.device, dtype=vlm_model.dtype)

    input_lengths = [len(ids) for ids in model_inputs.input_ids]
    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    trimmed = [ids[len:] for ids, len in zip(generated_ids, input_lengths)]
    response = vlm_processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant", 1)[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>", 1)[0]
    return response.strip()


def build_clip_cache_for_dataset(
    clip_dataset,
    cache_root: Path,
    prompt_hash: str,
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_new_tokens: int = 256,
    device: str = "cuda:0",
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 4,
    num_workers: int = 2,
):
    """
    Build CLIP VLM feature cache for the given dataset.
    Called automatically if cache is missing during training initialization.
    """
    is_main_process = rank == 0

    if is_main_process:
        print(f"🔄 Building CLIP VLM feature cache (this may take a while)...")
        print(f"   Cache directory: {cache_root}")
        print(f"   Prompt hash: {prompt_hash}")
        print(f"   VLM model: {vlm_model_name}")
        print(f"   Batch size: {batch_size} (per GPU)")
        print(f"   Workers: {num_workers} (per GPU)")

    # Load VLM model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if is_main_process:
        print(f"⏳ Loading VLM model on {device}...")

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

    if is_main_process:
        print(f"✅ VLM model loaded on {device}")

    # Create cache manager
    cache_manager = VLACacheManager(cache_dir=str(cache_root))

    # Create dataloader for cache building
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        clip_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    ) if world_size > 1 else None

    def collate_fn_cache(batch):
        """Collate function that keeps samples as list for batch processing"""
        return batch

    dataloader = DataLoader(
        clip_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler else False,
        num_workers=num_workers,
        collate_fn=collate_fn_cache
    )

    # Process samples
    if is_main_process:
        pbar = tqdm(total=len(dataloader), desc=f"Rank {rank} caching")
        print(f"📊 Processing {len(dataloader)} samples...")

    cached_count = 0
    skipped_count = 0

    for batch in dataloader:
        # Process each sample in the batch
        for sample in batch:
            images = sample.get("hand_eye_image")  # Now a list of images
            episode_id = sample.get("episode_id")
            vlm_idx = sample.get("vlm_idx")

            if vlm_idx is None or episode_id is None or images is None:
                skipped_count += 1
                continue

            # Ensure images is a list
            if not isinstance(images, list):
                images = [images]

            # Extract task_name from episode
            task_name = "Unknown"
            for sub_dataset in clip_dataset.unified_dataset.datasets:
                if sub_dataset.data_dir.name == episode_id:
                    task_name = extract_task_name_from_episode_path(sub_dataset.data_dir)
                    break

            # Get formatted prompt and hash for this task
            formatted_prompt = get_formatted_clip_prompt(task_name)
            task_prompt_hash = get_clip_prompt_hash(task_name)

            # Check if cache already exists (using task-specific hash)
            if cache_manager.cache_exists(dataset_name=episode_id, vlm_idx=vlm_idx, prompt_hash=task_prompt_hash):
                skipped_count += 1
                continue

            # Generate features
            try:
                # Build content with multiple images
                content = []
                for img in images:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": formatted_prompt})

                messages = [{
                    "role": "user",
                    "content": content
                }]
                generation_text_input = vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                vision_input, _ = process_vision_info(messages)

                text_response = _generate_text_response_local(
                    vlm_model, vlm_processor, generation_text_input, vision_input, max_new_tokens
                )

                with torch.no_grad():
                    # 1. Image-only inference (extract all image tokens)
                    # Build content with multiple images
                    image_only_content = []
                    for img in images:
                        image_only_content.append({"type": "image", "image": img})
                    image_only_content.append({"type": "text", "text": ""})

                    image_only_messages = [{
                        "role": "user",
                        "content": image_only_content
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

                    # Extract image tokens (token ID 151655 for Qwen2.5-VL image_pad)
                    image_token_mask = (image_inputs['input_ids'] == 151655)
                    image_indices = torch.where(image_token_mask.squeeze(0))[0]
                    image_features = image_hidden_state[:, image_indices, :]

                    # 2. Text-only inference (guidance vector via mean pooling)
                    text_inputs = vlm_processor(
                        text=[text_response],
                        images=None,
                        padding=True,
                        return_tensors="pt"
                    ).to(device=vlm_model.device, dtype=vlm_model.dtype)

                    text_outputs = vlm_model(**text_inputs, output_hidden_states=True, return_dict=True)
                    text_hidden_state = text_outputs.hidden_states[-1]
                    guidance_vector = text_hidden_state.mean(dim=1)

                # 3. Save cache (tuple format)
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
                cached_count += 1

            except Exception as e:
                if is_main_process:
                    print(f"⚠️ Error processing {episode_id}_vlm{vlm_idx}: {e}")

        if is_main_process:
            pbar.update(1)

    if is_main_process:
        pbar.close()
        print(f"✅ Cache building complete: {cached_count} new, {skipped_count} skipped")

    # Clean up VLM model
    del vlm_model
    del vlm_processor
    torch.cuda.empty_cache()


def infer_cached_feature_spec(
    cache_root: Path,
    prompt_hash: str,
    return_none_if_missing: bool = False
) -> Optional[Tuple[int, int, torch.dtype]]:
    """
    Inspect the cached VLM feature directory and return
    (image_feature_dim, text_feature_dim, dtype).

    If return_none_if_missing=True, returns None if no cache found.
    Otherwise raises RuntimeError.

    Now supports tuple format: (image_features, guidance_vector)
    """
    prompt_dir = Path(cache_root) / prompt_hash
    if not prompt_dir.exists():
        if return_none_if_missing:
            return None
        raise RuntimeError(
            f"Cached VLM feature directory not found: {prompt_dir}. "
            "Cache will be built automatically during first training run."
        )

    candidate_files = sorted(prompt_dir.glob("*_vlm*.pt"))
    if not candidate_files:
        if return_none_if_missing:
            return None
        raise RuntimeError(
            f"No cache files found under {prompt_dir}. "
            "Cache will be built automatically during first training run."
        )

    for candidate in candidate_files:
        try:
            data = torch.load(candidate, map_location="cpu")
        except Exception:
            continue

        # New tuple format: (image_features, guidance_vector)
        if isinstance(data, tuple) and len(data) == 2:
            image_features, guidance_vector = data

            if (
                isinstance(image_features, torch.Tensor)
                and image_features.ndim >= 1
                and image_features.shape[-1] > 0
                and isinstance(guidance_vector, torch.Tensor)
                and guidance_vector.numel() > 0
            ):

                # image_features: (1, N_tokens, D) or (N_tokens, D)
                # guidance_vector: (1, D) or (D,)
                img_squeezed = image_features.squeeze()
                txt_squeezed = guidance_vector.squeeze()

                if img_squeezed.ndim >= 1 and txt_squeezed.ndim >= 1:
                    image_dim = img_squeezed.shape[-1]
                    text_dim = txt_squeezed.shape[-1]
                    dtype = img_squeezed.dtype
                    return image_dim, text_dim, dtype

        # Legacy dict format support (for backwards compatibility)
        if isinstance(data, dict) and 'image_embed' in data and 'text_embed_sequence' in data:
            image_embed = data['image_embed']
            text_embed_seq = data['text_embed_sequence']

            if isinstance(image_embed, torch.Tensor) and image_embed.numel() > 0 and \
               isinstance(text_embed_seq, torch.Tensor) and text_embed_seq.numel() > 0:

                img_squeezed = image_embed.squeeze()
                txt_squeezed = text_embed_seq.squeeze()

                if img_squeezed.ndim >= 1 and txt_squeezed.ndim >= 1:
                    image_dim = img_squeezed.shape[-1]
                    text_dim = txt_squeezed.shape[-1]
                    dtype = img_squeezed.dtype
                    return image_dim, text_dim, dtype

    if return_none_if_missing:
        return None

    raise RuntimeError(
        f"No valid cached VLM tensors (tuple or dict format) found under {prompt_dir}. "
        "Cache will be built automatically during first training run."
    )

# =====================================
# 0. Data Augmentation
# =====================================

class SensorAugmentation:
    """
    Sensor data augmentation for CLIP training.
    All augmentations have ≤10% probability (mostly off).
    """
    def __init__(self,
                 time_mask_ratio=0.1,
                 noise_std=0.005,
                 scale_range=(0.97, 1.03),
                 p_time_mask=0.05,
                 p_noise=0.07,
                 p_scale=0.10):
        self.time_mask_ratio = time_mask_ratio
        self.noise_std = noise_std
        self.scale_range = scale_range

        # 각 어그멘테이션 적용 확률 (모두 10% 이하)
        self.p_time_mask = p_time_mask
        self.p_noise = p_noise
        self.p_scale = p_scale

    def __call__(self, sensor_data):
        """
        Args:
            sensor_data: (T, C=1026) - distance features (1-1025) + force (1026)
        """
        augmented = sensor_data.clone()
        device = augmented.device

        # 1. Time masking (≤10% probability, default 5%)
        if np.random.random() < self.p_time_mask:
            T = augmented.shape[0]
            num_mask = int(T * self.time_mask_ratio)
            if num_mask > 0:
                mask_indices = torch.randperm(T, device=device)[:num_mask]
                augmented[mask_indices] = 0.0

        # 2. Gaussian noise (≤10% probability, default 7%)
        if np.random.random() < self.p_noise:
            noise = torch.randn_like(augmented, device=device) * self.noise_std
            # Force channel (last) gets slightly more noise
            noise[:, -1] *= 1.5
            augmented += noise

        # 3. Magnitude scaling (≤10% probability, default 10%)
        if np.random.random() < self.p_scale:
            scale = np.random.uniform(*self.scale_range)
            augmented *= scale

        return augmented

# =====================================
# 1. CLIP-Style Dataset
# =====================================

class SensorImageCLIPDataset(Dataset):
    """
    A wrapper dataset that provides (sensor_data, hand_eye_image) pairs
    for contrastive pre-training.

    Only includes samples that are >= target_found_timestamp for efficiency.
    Caches the filtered indices to speed up subsequent runs.
    """
    def __init__(self, unified_dataset: UnifiedVLADataset, vlm_annotations: dict = None,
                 use_augmentation: bool = True, cache_path: str = None, clip_cache_root: str = None,
                 mode: Literal["train", "cache_build"] = "train", force_on_the_fly: bool = False,
                 skip_cache_verification: bool = False): # Added new argument
        self.unified_dataset = unified_dataset
        self.force_on_the_fly = force_on_the_fly
        if self.force_on_the_fly and (not dist.is_initialized() or dist.get_rank() == 0):
            print("⚡ Forcing on-the-fly VLM inference. Existing cache will be ignored and overwritten.")
        self.hand_eye_view_keywords = ["View5"]  # Use View5 only
        self.vlm_annotations = vlm_annotations if vlm_annotations is not None else {}
        self.cache_path = cache_path # Cache for filtered indices
        self.mode = mode
        if self.mode not in ("train", "cache_build"):
            raise ValueError(f"Unsupported mode '{self.mode}'. Expected 'train' or 'cache_build'.")
        self.require_cached_features = self.mode != "cache_build"
        self._vlm_idx_cache: Dict[Tuple[str, int], Optional[int]] = {}
        self.skip_cache_verification = skip_cache_verification # Store new argument

        # Cache for VLM features
        self.clip_cache_manager = None
        self.clip_prompt_hash = None  # Will be task-specific
        self.task_to_prompt_hash = {}  # Map task_name -> prompt_hash
        if self.require_cached_features and not clip_cache_root:
            raise ValueError("clip_cache_root must be provided when mode!='cache_build'")
        if clip_cache_root and self.require_cached_features:
            self.clip_cache_manager = VLACacheManager(cache_dir=clip_cache_root)
            # Build task_name -> prompt_hash mapping
            for sub_dataset in self.unified_dataset.datasets:
                task_name = extract_task_name_from_episode_path(sub_dataset.data_dir)
                if task_name not in self.task_to_prompt_hash:
                    self.task_to_prompt_hash[task_name] = get_clip_prompt_hash(task_name)
            print(f"   ... Using CLIP VLM feature cache at: {clip_cache_root}")
            print(f"   ... Task-specific prompt hashes: {self.task_to_prompt_hash}")

        # Data augmentation
        self.use_augmentation = use_augmentation
        self.is_training = True  # Training mode by default
        if self.use_augmentation:
            self.sensor_aug = SensorAugmentation()

        # Pre-filter valid indices (samples >= target_found_timestamp)
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if is_main_process:
            print("📋 Filtering dataset for CLIP training (only samples >= target_found_timestamp)...")
        
        self.valid_indices, self.valid_sample_metadata = self._filter_valid_samples()
        
        if is_main_process:
            print(f"✓ Filtered: {len(self.valid_indices)}/{len(self.unified_dataset)} samples are valid ({len(self.valid_indices)/len(self.unified_dataset)*100:.1f}%)")

    def _filter_valid_samples(self):
        """
        Filter samples using per-dataset independent caching.
        Each dataset folder has its own cache file, allowing incremental updates.
        """
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        # Use per-dataset caching strategy with shared fallback between modes
        cache_root = Path(__file__).parent / "cache"
        primary_subdir = "clip_filter_indices" if self.require_cached_features else "clip_filter_indices_cache_build"
        fallback_subdir = "clip_filter_indices_cache_build" if self.require_cached_features else "clip_filter_indices"
        cache_dir = cache_root / primary_subdir
        fallback_cache_dir = cache_root / fallback_subdir

        if is_main_process:
            cache_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        def _load_cached_indices(base_dir: Path, episode: str):
            if base_dir is None:
                return None
            cache_path = base_dir / f"{episode}.json"
            if not cache_path.exists():
                return None
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return None

            if not isinstance(cached, dict) or "indices" not in cached:
                return None

            # Allow cross-mode reuse (train/cache_build produce identical indices)
            if cached.get("mode") not in ("train", "cache_build"):
                return None
            return cached

        valid_indices = []
        metadata_entries = []
        global_idx_offset = 0

        total_episodes = 0
        episodes_with_target = 0

        pbar_datasets = tqdm(self.unified_dataset.datasets, desc="   Filtering episodes", disable=not is_main_process)

        for sub_dataset in pbar_datasets:
            num_samples_in_episode = len(sub_dataset)
            if num_samples_in_episode == 0:
                global_idx_offset += num_samples_in_episode
                continue

            episode_id = sub_dataset.data_dir.name
            task_name = extract_task_name_from_episode_path(sub_dataset.data_dir)
            task_prompt_hash = self.task_to_prompt_hash.get(task_name) if self.task_to_prompt_hash else get_clip_prompt_hash(task_name)
            pbar_datasets.set_postfix(episode=episode_id, task=task_name)

            # Per-episode cache file
            episode_cache_path = cache_dir / f"{episode_id}.json"

            # Try to load from cache
            episode_valid_indices = None
            cached_data = _load_cached_indices(cache_dir, episode_id)
            if cached_data is None and fallback_cache_dir != cache_dir:
                cached_data = _load_cached_indices(fallback_cache_dir, episode_id)

            if cached_data is not None:
                cached_prompt_hash = cached_data.get("prompt_hash")
                # Check if cached prompt hash matches this task's hash
                if self.require_cached_features and cached_prompt_hash != task_prompt_hash:
                    episode_valid_indices = None
                else:
                    episode_valid_indices = cached_data["indices"]
                    # Conditionally skip cache verification
                    if not self.skip_cache_verification:
                        episode_valid_indices = self._ensure_cached_features_exist(
                            episode_valid_indices, sub_dataset, episode_id, is_main_process=is_main_process,
                            task_prompt_hash=task_prompt_hash
                        )
                    cached_data["indices"] = episode_valid_indices
                    cached_data["mode"] = self.mode
                    cached_data["prompt_hash"] = task_prompt_hash
                    cached_data["task_name"] = task_name
                    if is_main_process:
                        with open(episode_cache_path, 'w') as f:
                            json.dump(cached_data, f, indent=2)
                    if dist.is_initialized():
                        dist.barrier()

            # If cache miss, filter this episode
            if episode_valid_indices is None:
                total_episodes += 1

                # Strategy: Use last 15% of episode data (85% onwards)
                # Include all samples, cache will be built on-the-fly during training
                start_idx = int(num_samples_in_episode * 0.85)
                candidate_indices = list(range(start_idx, num_samples_in_episode))

                # Include ALL samples from candidate_indices (cache will be built on-the-fly if needed)
                episode_valid_indices = []
                for sample_idx_local in candidate_indices:
                    vlm_idx = self._get_vlm_idx_for_sample(sub_dataset, episode_id, sample_idx_local)
                    if vlm_idx is None:
                        continue
                    episode_valid_indices.append(sample_idx_local)

                # Save to cache (main process only)
                if is_main_process:
                    with open(episode_cache_path, 'w') as f:
                        json.dump({
                            "indices": episode_valid_indices,
                            "episode_id": episode_id,
                            "task_name": task_name,
                            "start_idx": start_idx,
                            "total_samples": num_samples_in_episode,
                            "candidate_samples": len(candidate_indices),
                            "valid_samples": len(episode_valid_indices),
                            "strategy": "last_20_percent_with_cache" if self.require_cached_features else "last_20_percent_all",
                            "mode": self.mode,
                            "prompt_hash": task_prompt_hash,
                        }, f)

                # Barrier to ensure all processes wait for rank 0 to write the cache
                if dist.is_initialized():
                    dist.barrier()

                if len(episode_valid_indices) > 0:
                    episodes_with_target += 1

            # Add global offset to episode indices + metadata for downstream caching
            for sample_idx_local in episode_valid_indices:
                global_sample_idx = global_idx_offset + sample_idx_local
                valid_indices.append(global_sample_idx)

                vlm_idx = self._get_vlm_idx_for_sample(sub_dataset, episode_id, sample_idx_local) if self.require_cached_features else None

                metadata_entries.append({
                    "global_idx": global_sample_idx,
                    "episode_id": episode_id,
                    "local_idx": sample_idx_local,
                    "vlm_idx": vlm_idx,
                })
            global_idx_offset += num_samples_in_episode

        # Print statistics
        if is_main_process and total_episodes > 0:
            print(f"\n📊 Filtering Statistics:")
            print(f"   Total episodes processed: {total_episodes}")
            print(f"   Strategy: Last 20% of each episode (80%+) with{' ' if self.require_cached_features else 'out '}CLIP VLM cache filtering")
            print(f"   Episodes with valid cached samples: {episodes_with_target}")
            print(f"   Total valid samples: {len(valid_indices)}")

        return valid_indices, metadata_entries

    def _ensure_cached_features_exist(
        self,
        episode_valid_indices: List[int],
        sub_dataset,
        episode_id: str,
        is_main_process: bool = False,
        task_prompt_hash: str = None,
    ) -> List[int]:
        """
        Revalidate cached indices loaded from disk and drop samples whose VLM cache file
        has been removed since the cache file was first created.
        """
        if (
            not episode_valid_indices
            or not self.require_cached_features
            or self.clip_cache_manager is None
            or not task_prompt_hash
        ):
            return episode_valid_indices

        validated = []
        dropped = 0
        for sample_idx_local in episode_valid_indices:
            vlm_idx = self._get_vlm_idx_for_sample(sub_dataset, episode_id, sample_idx_local)
            if vlm_idx is None:
                dropped += 1
                continue
            if self.clip_cache_manager.cache_exists(
                dataset_name=episode_id,
                vlm_idx=vlm_idx,
                prompt_hash=task_prompt_hash,
            ):
                validated.append(sample_idx_local)
            else:
                dropped += 1

        if dropped and is_main_process:
            print(
                f"   ... Removed {dropped} cached indices without VLM features for episode {episode_id}."
            )
        return validated

    def _get_vlm_idx_for_sample(self, sub_dataset, episode_id: str, sample_idx_local: int) -> Optional[int]:
        """
        Map a dataset-local index to its corresponding VLM cache index.
        Results are memoized per (episode_id, local_idx) to avoid recomputation.
        """
        key = (episode_id, sample_idx_local)
        if key in self._vlm_idx_cache:
            return self._vlm_idx_cache[key]

        vlm_idx = self._compute_vlm_idx(sub_dataset, sample_idx_local)
        self._vlm_idx_cache[key] = vlm_idx
        return vlm_idx

    def _compute_vlm_idx(self, sub_dataset, sample_idx_local: int) -> Optional[int]:
        reuse_count = getattr(sub_dataset, "vlm_reuse_count", 1)
        reuse_count = reuse_count if reuse_count and reuse_count > 0 else 1

        interval = getattr(sub_dataset, "vlm_interval", None)
        if isinstance(interval, (int, float)) and interval > 0:
            return int((sample_idx_local // reuse_count) * interval)

        if getattr(sub_dataset, "format", None) == "old":
            total_actions = len(getattr(sub_dataset, "actions", []))
            base = (sample_idx_local // reuse_count) * reuse_count
            if total_actions > 0:
                base = min(base, total_actions - 1)
            return int(base)

        try:
            sample = sub_dataset[sample_idx_local]
            vlm_idx = sample.get("vlm_idx")
            if vlm_idx is not None:
                return int(vlm_idx)
        except Exception:
            return None

        return None

    def _load_hand_eye_images(self, sample):
        """
        Load ONLY View5 hand-eye camera image for cache building mode.
        Returns a list with single PIL Image (View5 only).
        """
        # Try to get from sample["images"] first
        image_entries = sample.get("images") or []

        # Only look for View5
        view5_keyword = "View5"
        view5_entry = None

        def is_view5(entry_path: str):
            """Check if entry is View5"""
            return view5_keyword.lower() in entry_path.lower()

        for entry in image_entries:
            if entry is None:
                continue

            if isinstance(entry, Image.Image):
                # PIL image - use it as View5
                view5_entry = entry
                break

            entry_path = str(entry)
            if not entry_path:
                continue

            if is_view5(entry_path):
                view5_entry = entry_path
                break

        # Load View5 image
        loaded_images = []
        if view5_entry:
            if isinstance(view5_entry, Image.Image):
                loaded_images.append(view5_entry)
            else:
                loaded_images.append(Image.open(view5_entry).convert("RGB"))

        # If no View5 image found from sample["images"], try to construct path manually
        if not loaded_images:
            episode_id = sample.get("episode_id")
            vlm_idx = sample.get("vlm_idx")

            if episode_id and vlm_idx is not None:
                # Find the dataset containing this episode
                for sub_dataset in self.unified_dataset.datasets:
                    if sub_dataset.data_dir.name == episode_id:
                        data_dir = sub_dataset.data_dir

                        # Only try View5
                        possible_view_dirs = [
                            data_dir / "View5",  # New format: data_collection_*/View5
                            data_dir / "images" / "View5",  # Old format: episode_*/images/View5
                        ]

                        for view_dir in possible_view_dirs:
                            if view_dir.exists():
                                # Get all images in this directory
                                image_files = sorted(list(view_dir.glob("*.jpg")) + list(view_dir.glob("*.png")))
                                if image_files and vlm_idx < len(image_files):
                                    loaded_images.append(Image.open(image_files[vlm_idx]).convert("RGB"))
                                    break

                        if loaded_images:
                            break

        if not loaded_images:
            raise RuntimeError(
                f"No valid View5 image found. "
                f"episode_id={sample.get('episode_id')}, vlm_idx={sample.get('vlm_idx')}"
            )

        return loaded_images

    def train(self):
        """Enable augmentation for training"""
        self.is_training = True

    def eval(self):
        """Disable augmentation for validation"""
        self.is_training = False

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map to actual index in unified_dataset
        actual_idx = self.valid_indices[idx]
        metadata = self.valid_sample_metadata[idx] if self.valid_sample_metadata else {}

        # Get the full data sample from the underlying dataset
        try:
            sample = self.unified_dataset[actual_idx]
        except (FileNotFoundError, IndexError) as e:
            print(f"Warning: Skipping index {actual_idx} due to error: {e}")
            sample = self.unified_dataset[self.valid_indices[0]]

        sensor_data = sample["sensor_data"]
        timestamp = sample.get("timestamp")

        episode_id = metadata.get("episode_id")
        if episode_id is None:
            episode_id = sample.get("episode_id")

        vlm_idx = metadata.get("vlm_idx")
        if vlm_idx is None:
            sample_vlm_idx = sample.get("vlm_idx")
            if sample_vlm_idx is not None:
                vlm_idx = int(sample_vlm_idx)
        else:
            vlm_idx = int(vlm_idx)
        hand_eye_representation = None
        vlm_cache_key = (episode_id, vlm_idx) if episode_id and vlm_idx is not None else None

        if self.require_cached_features:
            # Extract task_name for this episode and get corresponding prompt_hash
            task_name = None
            for sub_dataset in self.unified_dataset.datasets:
                if sub_dataset.data_dir.name == episode_id:
                    task_name = extract_task_name_from_episode_path(sub_dataset.data_dir)
                    break

            if task_name is None:
                raise RuntimeError(f"Could not find task_name for episode_id={episode_id}")

            task_prompt_hash = self.task_to_prompt_hash.get(task_name)
            if not task_prompt_hash:
                task_prompt_hash = get_clip_prompt_hash(task_name)

            # Load dictionary of features from CLIP VLM feature cache
            if not self.clip_cache_manager or episode_id is None or vlm_idx is None or not task_prompt_hash:
                raise RuntimeError(
                    f"CLIP cache manager not initialized or missing metadata. "
                    f"episode_id={episode_id}, vlm_idx={vlm_idx}, task={task_name}, prompt_hash={task_prompt_hash}"
                )
            if self.force_on_the_fly:
                cached_features = None
            else:
                cached_features = self._load_cached_vlm_feature(episode_id, vlm_idx, task_prompt_hash)

            # If cache exists, use it
            if cached_features is not None:
                vlm_image_features = cached_features['image_features']  # (N_tokens, D)
                vlm_guidance_vector = cached_features['guidance_vector']  # (D,)
            else:
                # Cache doesn't exist, load images for on-the-fly VLM inference
                hand_eye_representation = self._load_hand_eye_images(sample)
                vlm_image_features = hand_eye_representation  # Pass list of images
                vlm_guidance_vector = None  # Will be computed on-the-fly
        else:
            # In cache_build mode, we load the raw images (all views)
            hand_eye_representation = self._load_hand_eye_images(sample)
            # Placeholders for the items that will be created from these images
            vlm_image_features = hand_eye_representation # Pass list of images to be processed
            vlm_guidance_vector = torch.empty(0) # Not used in cache build mode

        timestamp = float(timestamp if timestamp is not None else 0.0)
        vlm_description = self.vlm_annotations.get(episode_id, {}).get(str(timestamp), "no_vlm_description")

        # Apply sensor augmentation (only during training)
        if self.use_augmentation and self.is_training:
            sensor_data = self.sensor_aug(sensor_data)

        if self.require_cached_features:
            # Check if we need on-the-fly VLM inference (guidance_vector is None means no cache)
            needs_vlm_inference = (vlm_guidance_vector is None)

            return {
                "sensor_data": sensor_data,
                "vlm_image_features": vlm_image_features,
                "vlm_guidance_vector": vlm_guidance_vector,
                "vlm_description": vlm_description,
                "episode_id": episode_id,
                "timestamp": timestamp,
                "vlm_idx": vlm_idx,
                "vlm_cache_key": vlm_cache_key,
                "needs_vlm_inference": needs_vlm_inference,
                "task_name": task_name,
                "task_prompt_hash": task_prompt_hash,
            }
        else: # cache_build mode
            return {
                "sensor_data": sensor_data,
                "hand_eye_image": vlm_image_features, # Pass the PIL image
                "vlm_description": vlm_description,
                "episode_id": episode_id,
                "timestamp": timestamp,
                "vlm_idx": vlm_idx,
                "vlm_cache_key": vlm_cache_key,
            }

    def _load_cached_vlm_feature(self, episode_id: str, vlm_idx: int, task_prompt_hash: str) -> Dict[str, torch.Tensor]:
        vlm_feature = self.clip_cache_manager.load_cache(
            dataset_name=episode_id,
            vlm_idx=vlm_idx,
            prompt_hash=task_prompt_hash
        )

        # Return None if cache doesn't exist (will be built on-the-fly)
        if vlm_feature is None:
            return None

        # Handle tuple format: (image_features, guidance_vector)
        if isinstance(vlm_feature, tuple) and len(vlm_feature) == 2:
            image_features, guidance_vector = vlm_feature
            return {
                'image_features': image_features.squeeze(0) if image_features.dim() > 2 else image_features,
                'guidance_vector': guidance_vector.squeeze(0) if guidance_vector.dim() > 1 else guidance_vector
            }

        # Legacy dict format support
        if isinstance(vlm_feature, dict):
            return {
                'image_features': vlm_feature.get('image_embed', torch.empty(0)),
                'guidance_vector': vlm_feature.get('text_embed_sequence', torch.empty(0))
            }

        raise RuntimeError(f"Unexpected cache format for episode={episode_id}, vlm_idx={vlm_idx}")

def process_images_with_vlm(images, prompt, vlm_model, vlm_processor, device, max_new_tokens=256):
    """
    Helper function to perform on-the-fly VLM inference.
    This logic is extracted from `build_clip_cache_for_dataset`.
    """
    # Build content with multiple images
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    generation_text_input = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    vision_input, _ = process_vision_info(messages)

    text_response = _generate_text_response_local(
        vlm_model, vlm_processor, generation_text_input, vision_input, max_new_tokens
    )

    with torch.no_grad():
        # 1. Image-only inference (extract all image tokens)
        image_only_content = []
        for img in images:
            image_only_content.append({"type": "image", "image": img})
        image_only_content.append({"type": "text", "text": ""})

        image_only_messages = [{"role": "user", "content": image_only_content}]
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

        image_token_mask = (image_inputs['input_ids'] == 151655)
        image_indices = torch.where(image_token_mask.squeeze(0))[0]
        image_features = image_hidden_state[:, image_indices, :]

        # 2. Text-only inference (guidance vector via mean pooling)
        text_inputs = vlm_processor(
            text=[text_response],
            images=None,
            padding=True,
            return_tensors="pt"
        ).to(device=vlm_model.device, dtype=vlm_model.dtype)

        text_outputs = vlm_model(**text_inputs, output_hidden_states=True, return_dict=True)
        text_hidden_state = text_outputs.hidden_states[-1]
        guidance_vector = text_hidden_state.mean(dim=1)

    image_features = image_features.squeeze(0) if image_features.dim() > 2 else image_features
    guidance_vector = guidance_vector.squeeze(0) if guidance_vector.dim() > 1 else guidance_vector
    
    return image_features, guidance_vector

def clip_collate_fn(batch, window_size, vlm_model=None, vlm_processor=None, clip_cache_manager=None, device=None):
    """
    Robust collate function that pads sensor data and handles on-the-fly VLM inference.
    - If cache exists, use it
    - If cache doesn't exist, perform VLM inference and save to cache
    """
    sensor_tensors = []
    vlm_image_features_list = []
    vlm_guidance_vectors = []
    hand_eye_images = []

    vlm_descriptions = []
    episode_ids = []
    timestamps = []
    vlm_cache_keys = []

    is_training_mode = "vlm_image_features" in batch[0]

    # Track on-the-fly caching stats
    needs_inference_count = 0
    used_cache_count = 0

    for sample in batch:
        sensor = sample["sensor_data"]
        # Truncate or pad the sensor data to the fixed window size
        if sensor.shape[0] > window_size:
            sensor = sensor[:window_size, :]
        elif sensor.shape[0] < window_size:
            pad = torch.zeros(
                (window_size - sensor.shape[0], sensor.shape[1]),
                dtype=sensor.dtype,
                device=sensor.device,
            )
            sensor = torch.cat([sensor, pad], dim=0)
        sensor_tensors.append(sensor)

        if is_training_mode:
            needs_vlm_inference = sample.get("needs_vlm_inference", False)

            if needs_vlm_inference and vlm_model is not None:
                # Perform on-the-fly VLM inference
                needs_inference_count += 1
                images = sample["vlm_image_features"]  # List of PIL images
                task_name = sample.get("task_name", "unknown")
                episode_id = sample.get("episode_id")
                vlm_idx = sample.get("vlm_idx")
                task_prompt_hash = sample.get("task_prompt_hash")

                # Run VLM inference
                prompt = get_formatted_clip_prompt(task_name)
                try:
                    image_features, guidance_vector = process_images_with_vlm(
                        images, prompt, vlm_model, vlm_processor, device
                    )

                    # Save to cache for next epoch
                    if clip_cache_manager and episode_id and vlm_idx is not None and task_prompt_hash:
                        features_to_cache = (
                            image_features.detach().to("cpu", dtype=torch.float16),
                            guidance_vector.detach().to("cpu", dtype=torch.float16)
                        )
                        clip_cache_manager.save_cache_tuple(
                            dataset_name=episode_id,
                            vlm_idx=vlm_idx,
                            prompt_hash=task_prompt_hash,
                            features_tuple=features_to_cache
                        )

                    vlm_image_features_list.append(image_features)
                    vlm_guidance_vectors.append(guidance_vector)
                except Exception as e:
                    print(f"Warning: VLM inference failed for {episode_id}/{vlm_idx}: {e}")
                    # Use dummy tensors as fallback
                    vlm_image_features_list.append(torch.zeros(1, 3584, device=device))
                    vlm_guidance_vectors.append(torch.zeros(3584, device=device))
            else:
                # Use cached features
                used_cache_count += 1
                vlm_image_features_list.append(sample["vlm_image_features"])
                vlm_guidance_vectors.append(sample["vlm_guidance_vector"])
        else:
            hand_eye_images.append(sample["hand_eye_image"])

        vlm_descriptions.append(sample["vlm_description"])
        episode_ids.append(sample["episode_id"])
        timestamps.append(sample["timestamp"])
        vlm_cache_keys.append(sample.get("vlm_cache_key"))

    collated_batch = {
        "sensor_data": torch.stack(sensor_tensors),
        "vlm_description": vlm_descriptions,
        "episode_ids": episode_ids,
        "timestamps": timestamps,
        "vlm_cache_keys": vlm_cache_keys,
    }

    if is_training_mode:
        # 모두 float16로 저장되어 있어도, 여기서 일괄 float32로 맞춰도 좋고
        # (autocast가 bf16로 내릴 것) 바로 float32로 맞추는 편이 안전하다.
        # Also ensure all tensors are on the same device (cached features might be on CPU)
        vlm_image_features_list = [x.to(device=device, dtype=torch.float32) for x in vlm_image_features_list]
        vlm_guidance_vectors    = [x.to(device=device, dtype=torch.float32) for x in vlm_guidance_vectors]

        # Check for dimension consistency
        if vlm_image_features_list:
            feature_dims = [x.shape[-1] for x in vlm_image_features_list]
            guidance_dims = [x.shape[-1] for x in vlm_guidance_vectors]

            if len(set(feature_dims)) > 1 or len(set(guidance_dims)) > 1:
                raise RuntimeError(
                    f"Dimension mismatch in batch! "
                    f"Image feature dims: {set(feature_dims)}, "
                    f"Guidance vector dims: {set(guidance_dims)}. "
                    f"This indicates cache files were generated with different VLM models. "
                    f"Please regenerate cache with a consistent model or filter incompatible caches."
                )

        collated_batch["vlm_image_features"] = torch.nn.utils.rnn.pad_sequence(
            vlm_image_features_list, batch_first=True, padding_value=0.0
        )  # (B, max_N_tokens, D)
        collated_batch["vlm_guidance_vector"] = torch.stack(vlm_guidance_vectors)  # (B, D)

        # Add caching stats
        collated_batch["cache_stats"] = {
            "needs_inference": needs_inference_count,
            "used_cache": used_cache_count,
        }
    else:
        collated_batch["hand_eye_image"] = hand_eye_images

    return collated_batch


# =====================================
# 2. CLIP Model Definition
# =====================================

# 본 스크립트 내 CLIPModel 정의부 전면 교체

class CLIPModel(nn.Module):
    """
    (MODIFIED FOR SIZE)
    0) Projects high-dim VLM features down to an intermediate dimension.
    1) guidance(text) → image tokens cross-attn
    2) image tokens 전역 풀링(TokenAttnPool)
    3) fused_vlm ⊕ img_global → joint proj
    4) sensor seq(65×Ds) → 테일 가중 풀링(정렬) → proj
    5) (선택) image-only / text-only proj 추가 반환  ← 삼자 대조용
    """
    def __init__(
        self,
        sensor_encoder,
        sensor_output_dim: int,
        image_embedding_dim: int,
        text_embedding_dim: int,
        projection_dim: int = 512,
        intermediate_vlm_dim: int = 2048, # New parameter for size reduction, targeting ~500MB
        tail_bias: float = 1.0,
    ):
        super().__init__()
        self.sensor_encoder = sensor_encoder
        self.tail_bias = tail_bias

        # --- Start of modifications for size reduction ---
        # Project high-dim VLM features to a smaller intermediate dimension
        self.image_feature_proj = nn.Linear(image_embedding_dim, intermediate_vlm_dim)
        self.text_feature_proj = nn.Linear(text_embedding_dim, intermediate_vlm_dim)
        
        # Now, all subsequent layers will use `intermediate_vlm_dim`
        vlm_dim = intermediate_vlm_dim
        # --- End of modifications ---

        # guidance(query: text) ↔ image tokens(key/value: image)
        self.vlm_fusion_attention = nn.MultiheadAttention(
            embed_dim=vlm_dim,   # query dim
            num_heads=8,
            dropout=0.1,
            batch_first=True,
            kdim=vlm_dim,       # key dim
            vdim=vlm_dim        # value dim
        )

        # Sensor projection
        self.sensor_projection = nn.Sequential(
            nn.Linear(sensor_output_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim)
        )
        nn.utils.weight_norm(self.sensor_projection[0])

        # Image token global pooling + joint projection
        self.image_token_pool = TokenAttnPool(vlm_dim, heads=8, max_tokens=2048)
        self.vlm_joint_proj = nn.Sequential(
            nn.Linear(vlm_dim + vlm_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim)
        )
        nn.utils.weight_norm(self.vlm_joint_proj[0])

        # (선택) image-only / text-only projection heads  ← 3자 대조용
        self.img_only_proj = nn.Sequential(
            nn.Linear(vlm_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim)
        )
        nn.utils.weight_norm(self.img_only_proj[0])

        self.txt_only_proj = nn.Sequential(
            nn.Linear(vlm_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim)
        )
        nn.utils.weight_norm(self.txt_only_proj[0])

        # 외부 LN
        self.sensor_norm = nn.LayerNorm(projection_dim)
        self.vlm_norm    = nn.LayerNorm(projection_dim)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / SIGLIP_TEMPERATURE)))

    def _cast_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.device != ref.device or x.dtype != ref.dtype:
            x = x.to(device=ref.device, dtype=ref.dtype, non_blocking=True)
        return x

    @staticmethod
    def _tail_weighted_pool(seq_feat: torch.Tensor, tail_bias: float=1.0) -> torch.Tensor:
        """
        seq_feat: (B,T,Ds). 뒤로 갈수록 큰 가중치를 주는 선형 마스크.
        tail_bias=1.0이면 0.1→1.0 선형. (필요시 조정)
        """
        B, T, Ds = seq_feat.shape
        # 0.1 ~ (0.1+tail_bias*0.9) 사이 선형 증가
        hi = 0.1 + 0.9 * tail_bias
        w = torch.linspace(0.1, hi, steps=T, device=seq_feat.device, dtype=seq_feat.dtype).view(1,T,1)
        out = (seq_feat * w).sum(dim=1) / w.sum(dim=1)
        return out  # (B,Ds)

    def forward(self, sensor_data, vlm_image_features, vlm_guidance_vector):
        """
        sensor_data:        (B, T, C)
        vlm_image_features: (B, N_tokens, D_img)
        vlm_guidance_vector:(B, D_text)
        Returns:
            sensor_embedding: (B, D_proj)
            vlm_joint_emb:    (B, D_proj)
            vlm_img_only_emb: (B, D_proj)   # ← 추가
            vlm_txt_only_emb: (B, D_proj)   # ← 추가
            logit_scale:      scalar parameter
        """
        # 1) Sensor encode (시퀀스 + 전역)
        seq_feat, sensor_global = self.sensor_encoder(sensor_data, return_sequence=True)  # (B,T,Ds), (B,D_sensor)

        # 2) dtype/device 정렬
        vlm_image_features = self._cast_to(vlm_image_features, sensor_global)
        vlm_guidance_vector = self._cast_to(vlm_guidance_vector, sensor_global)

        # --- Start of modifications for size reduction ---
        # Project down to intermediate dimension
        vlm_image_features = self.image_feature_proj(vlm_image_features)
        vlm_guidance_vector = self.text_feature_proj(vlm_guidance_vector)
        # --- End of modifications ---

        # 3) PAD 마스크 (0.0으로 패딩된 토큰)
        key_padding_mask = (vlm_image_features.abs().sum(dim=-1) == 0)  # (B, N), bool

        # 4) guidance(query) → image tokens(key/value) cross-attn
        guidance_query = vlm_guidance_vector.unsqueeze(1)  # (B,1,D_text)
        fused_vlm_features, _ = self.vlm_fusion_attention(
            query=guidance_query,
            key=vlm_image_features,
            value=vlm_image_features,
            need_weights=False,
            key_padding_mask=key_padding_mask
        )
        fused_vlm = fused_vlm_features.squeeze(1)  # (B, D_text)

        # 5) 이미지 토큰 전역 풀링
        img_global = self.image_token_pool(vlm_image_features, key_padding_mask)  # (B, D_img)

        # 6) 결합 → joint projection
        vlm_joint = torch.cat([fused_vlm, img_global], dim=-1)  # (B, D_text + D_img)
        vlm_joint_emb  = F.normalize(self.vlm_norm(self.vlm_joint_proj(vlm_joint)), dim=-1)

        # 7) image-only / text-only proj  ← 3자 대조
        vlm_img_only_emb = F.normalize(self.vlm_norm(self.img_only_proj(img_global)), dim=-1)
        vlm_txt_only_emb = F.normalize(self.vlm_norm(self.txt_only_proj(vlm_guidance_vector)), dim=-1)

        # 8) Sensor: 시계열 → 테일 가중 풀링(정렬) → proj
        sensor_aligned = self._tail_weighted_pool(seq_feat, self.tail_bias)          # (B,Ds)
        sensor_emb = F.normalize(self.sensor_norm(self.sensor_projection(sensor_global if torch.isnan(sensor_aligned).any() else sensor_aligned)), dim=-1)

        # Return scale value (not parameter) to avoid DDP tracking issues
        scale = self.logit_scale.exp()
        return sensor_emb, vlm_joint_emb, vlm_img_only_emb, vlm_txt_only_emb, scale


# =====================================
# 3. SigLIP-Style Contrastive Loss
# =====================================

class TokenAttnPool(nn.Module):
    def __init__(self, d, heads=8, max_tokens=1024):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_tokens, d) / d**0.5)
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj = nn.Linear(d, d)

    def forward(self, x, key_padding_mask):
        B, N, D = x.shape
        pos = self.pos[:, :N, :]
        x = x + pos
        q = x.mean(dim=1, keepdim=True)          # 글로벌 쿼리 1개
        out, _ = self.mha(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.proj(out.squeeze(1))         # (B, D)

def siglip_loss(a: torch.Tensor, b: torch.Tensor, scale=None):
    # scale is already exp() of logit_scale, computed in forward pass
    if scale is None:
        scale = 1.0 / SIGLIP_TEMPERATURE
    logits = scale * (a @ b.T)
    B = logits.size(0)
    labels = torch.eye(B, device=logits.device, dtype=logits.dtype)
    loss_a = F.binary_cross_entropy_with_logits(logits,   labels)
    loss_b = F.binary_cross_entropy_with_logits(logits.T, labels)
    return 0.5 * (loss_a + loss_b)

def tri_clip_loss(sensor_emb, vlm_joint, vlm_img, vlm_txt, scale, w_joint=1.0, w_img=0.5, w_txt=0.5):
    main = siglip_loss(sensor_emb, vlm_joint, scale)
    img  = siglip_loss(sensor_emb, vlm_img,   scale)
    txt  = siglip_loss(sensor_emb, vlm_txt,   scale)
    return w_joint*main + w_img*img + w_txt*txt, (main, img, txt)



# =====================================
# 4. Main Training Function
# =====================================

@torch.no_grad()
def _all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """Validation 전용: 모든 rank에서 텐서를 모으되 grad 없음."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    world = dist.get_world_size()
    buf = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(buf, x.contiguous())
    return torch.cat(buf, dim=0)


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """
    Training용: all_gather 하되, 현재 rank의 텐서만 grad 경로 유지.
    다른 rank 텐서는 detach 상태로 합친다.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    world = dist.get_world_size()
    rank = dist.get_rank()

    # 먼저 detach로 모은 다음
    buf = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(buf, x.detach().contiguous())
    # 내 자리는 원본(grad 유지)로 교체
    buf[rank] = x
    return torch.cat(buf, dim=0)


def main(args):
    # DDP Setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = local_rank == 0
    device_props = torch.cuda.get_device_properties(local_rank)
    device_total_gb = device_props.total_memory / (1024 ** 3)

    if is_main_process:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")

    # --- VLM Loading ---
    vlm_model = None
    vlm_processor = None
    vlm_device = device  # Set device for all ranks

    if not args.cache_only_mode:
        if is_main_process:
            print(f"⏳ Attempting to load VLM model on all {world_size} ranks for hybrid on-the-fly caching...")

        try:
            vlm_processor = AutoProcessor.from_pretrained(args.vlm_model, trust_remote_code=True)
            vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.vlm_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=vlm_device,
                attn_implementation="flash_attention_2"
            )
            vlm_model.eval()
            disable_generation_temperature(vlm_model)

            # Freeze VLM parameters to prevent gradient calculation
            for param in vlm_model.parameters():
                param.requires_grad = False

            if is_main_process:
                print("✅ VLM model loaded successfully and frozen. Hybrid caching is enabled.")
        except Exception as e:
            if is_main_process:
                print(f"⚠️ WARNING: Failed to load VLM model. Training will proceed in cache-only mode. Error: {e}")
            # Ensure all ranks have None if loading fails
            vlm_model = None
            vlm_processor = None
    else:
        if is_main_process:
            print("INFO: Running in --cache_only_mode. VLM model will not be loaded.")


    clip_cache_root = Path(args.cache_root) / "clip_vlm_features"

    # Note: We'll check cache spec using a representative task's prompt hash
    # All tasks use the same VLM model, so feature dims should be identical
    # We'll use a dummy task name to check if any cache exists
    representative_prompt_hash = None
    cache_spec = None

    # Try to find any existing cache folder to infer spec
    if clip_cache_root.exists():
        for cache_dir in clip_cache_root.iterdir():
            if cache_dir.is_dir():
                representative_prompt_hash = cache_dir.name
                cache_spec = infer_cached_feature_spec(clip_cache_root, representative_prompt_hash, return_none_if_missing=True)
                if cache_spec is not None:
                    break

    if cache_spec is None:
        if is_main_process:
            print("⚠️ CLIP VLM feature cache not found. Training will skip uncached samples (no VLM loaded).")
        # Use safe defaults that match Qwen2.5-VL-3B; actual cached dims will be used per-sample.
        image_embed_dim = 3584
        text_embed_dim = 3584
        cached_feature_dtype = torch.float16
        needs_cache_build = False
    else:
        image_embed_dim, text_embed_dim, cached_feature_dtype = cache_spec
        needs_cache_build = False
        if is_main_process:
            print(f"✅ Cache found. Inferred: Image Dim={image_embed_dim}, Text Dim={text_embed_dim}, DType={cached_feature_dtype}")


    # If forcing on-the-fly, override the feature spec with the config from the loaded VLM
    if args.force_on_the_fly and vlm_model is not None:
        # For Qwen2.5-VL, both image and text features are projected to the text model's hidden size
        embed_dim = vlm_model.config.text_config.hidden_size
        image_embed_dim = embed_dim
        text_embed_dim = embed_dim
        if is_main_process:
            print(f"INFO: Overriding feature spec for on-the-fly mode.")
            print(f"      New Spec: Image Dim={image_embed_dim}, Text Dim={text_embed_dim}")


    # Sensor Encoder Setup will be done after cache building (if needed)
    # to ensure correct dimensions

    # Dataset and DataLoader
    if is_main_process: 
        print("Creating dataset...")

    # Load VLM annotations (on all processes)
    annotations = {}
    annotation_path = Path(args.annotation_path)
    if annotation_path.exists():
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        if is_main_process:
            print(f"Loaded {len(annotations)} VLM annotations from {annotation_path}")
    elif is_main_process:
        print(f"Warning: Annotation file not found at {annotation_path}. Proceeding without VLM-based weighting.")

    # Use per-dataset independent caching for unified dataset
    # This allows reusing cache when adding new datasets
    if is_main_process:
        print("Using per-dataset independent caching strategy")

    # NOTE: With task-specific prompts, each task (Red_point, White_point, etc.) now has its own
    # prompt hash. The SensorImageCLIPDataset handles task-specific prompt hashing internally
    # via task_to_prompt_hash mapping. If you edit CLIP_PROMPT_TEXT, regenerate all caches.
    unified_dataset = create_unified_dataloader(
        new_dataset_paths=args.new_dataset_paths,
        old_dataset_patterns=args.old_dataset_patterns,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        return_dataset=True,
        use_cache=False,  # Disable UnifiedVLADataset cache loading (CLIP uses its own task-specific cache)
        cache_root=str(clip_cache_root),  # Align unified dataset cache root with CLIP cache
        skip_dataset_stats=args.skip_dataset_stats,
    )

    # Create training dataset with on-the-fly caching enabled
    # No need to pre-build cache - collate function handles on-the-fly inference & caching
    clip_dataset = SensorImageCLIPDataset(
        unified_dataset,
        vlm_annotations=annotations,
        cache_path=None,  # Use per-episode caching instead of global cache
        clip_cache_root=str(clip_cache_root),
        mode="train",
        force_on_the_fly=args.force_on_the_fly,
        skip_cache_verification=args.skip_cache_verification # Pass the new argument
    )

    # Now initialize model with correct dimensions
    if is_main_process:
        print("Initializing Sensor Encoder and CLIP Model...")

    # Lightweight configuration to reduce model size and set 9:1 force/dist ratio
    force_hidden_dim_light = 48  # Approx. 10% of 512, multiple of 16
    dist_hidden_dim_light = 128
    num_transformer_layers_light = 1
    transformer_dim_light = 256

    if is_main_process:
        print(f"INFO: Using lightweight SensorEncoder configuration.")
        print(f"      - output_dim: {args.sensor_output_dim} (Force: {force_hidden_dim_light}, Dist: {args.sensor_output_dim - force_hidden_dim_light})")
        print(f"      - dist_hidden_dim (Conv): {dist_hidden_dim_light}")
        print(f"      - num_transformer_layers: {num_transformer_layers_light}")
        print(f"      - transformer_dim: {transformer_dim_light}")

    sensor_encoder = ForceAwareSensorEncoder(
        temporal_length=args.sensor_window_size,
        output_dim=args.sensor_output_dim,
        dist_hidden_dim=dist_hidden_dim_light,
        force_hidden_dim=force_hidden_dim_light,
        use_transformer=True,
        num_transformer_layers=num_transformer_layers_light,
        transformer_dim=transformer_dim_light
    )
    if args.sensor_precision == "bf16":
        sensor_encoder.to(device, dtype=torch.bfloat16)
    else:
        sensor_encoder.to(device)
    force_bn_fp32_(sensor_encoder)

    # CLIP Model & Optimizer
    clip_model = CLIPModel(
        sensor_encoder=sensor_encoder,
        sensor_output_dim=args.sensor_output_dim,
        image_embedding_dim=image_embed_dim,
        text_embedding_dim=text_embed_dim,
        projection_dim=args.embedding_dim,
        intermediate_vlm_dim=args.intermediate_vlm_dim, # Pass the new argument
    ).to(device)

    # Print model summary before wrapping with DDP
    if not dist.is_initialized() or dist.get_rank() == 0:
        # The CLIPModel now contains the sensor_encoder, so we summarize it as a whole.
        print_model_summary(clip_model, "Trainable CLIP Model (includes Sensor Encoder)")
        
        trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
        trainable_size_mb = trainable_params * 4 / (1024 ** 2)

        print("="*50)
        print("ESTIMATED CHECKPOINT SIZE")
        print("="*50)
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Estimated Model Weights Size: {trainable_size_mb:.2f} MB")
        print(f"  Estimated Checkpoint Size (Weights + Optimizer): {trainable_size_mb * 3:.2f} MB")
        print("="*50 + "\n")

    clip_model = DDP(
        clip_model,
        device_ids=[local_rank],
        find_unused_parameters=args.find_unused_parameters,
    )
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=args.learning_rate)

    if is_main_process:
        print("✅ Model initialized successfully.")

    # Split dataset
    val_size = int(len(clip_dataset) * args.val_split)
    train_size = len(clip_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(clip_dataset, [train_size, val_size])

    if is_main_process:
        print(f"Dataset created with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    collate_fn_with_window = partial(
        clip_collate_fn,
        window_size=args.sensor_window_size,
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
        clip_cache_manager=clip_dataset.clip_cache_manager,
        device=vlm_device
    )

    # If VLM model is loaded, disable workers and pin_memory to avoid CUDA conflicts in subprocesses.
    if vlm_model is not None:
        num_workers_to_use = 0
        pin_memory_flag = False
        if is_main_process:
            print("INFO: VLM model is loaded. Using num_workers=0 and pin_memory=False to prevent CUDA multiprocessing errors.")
    else:
        # Fallback to original logic for cache-only training
        num_workers_to_use = args.num_workers
        pin_memory_flag = False
        if is_main_process:
            print("INFO: VLM model not loaded. Using standard dataloader settings (num_workers > 0) but disabling pin_memory to avoid CUDA tensor errors.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_workers_to_use,
        collate_fn=collate_fn_with_window,
        pin_memory=pin_memory_flag,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, num_workers=num_workers_to_use,
        collate_fn=collate_fn_with_window, pin_memory=pin_memory_flag
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    # Checkpoint Loading
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        if is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        model_to_load = clip_model.module if isinstance(clip_model, DDP) else clip_model
        
        # Check for architecture mismatch by comparing state dicts
        ckpt_state_dict = checkpoint.get('model_state_dict', {})
        current_state_dict = model_to_load.state_dict()
        
        # Filter out layers that have different shapes
        new_state_dict = {}
        incompatible_keys = []
        for key, ckpt_param in ckpt_state_dict.items():
            if key in current_state_dict:
                current_param = current_state_dict[key]
                if ckpt_param.shape == current_param.shape:
                    new_state_dict[key] = ckpt_param
                else:
                    incompatible_keys.append(key)
            else:
                # Key from checkpoint not in current model
                incompatible_keys.append(key)

        model_to_load.load_state_dict(new_state_dict, strict=False)
        
        if is_main_process:
            if incompatible_keys:
                print("⚠️ WARNING: Architecture mismatch detected. Some layers were not loaded from checkpoint:")
                for key in incompatible_keys:
                    ckpt_shape = ckpt_state_dict[key].shape if key in ckpt_state_dict else 'N/A'
                    curr_shape = current_state_dict[key].shape if key in current_state_dict else 'N/A'
                    print(f"  - Layer '{key}': Checkpoint shape {ckpt_shape}, Model shape {curr_shape}")
            else:
                print("   Model weights loaded successfully.")

        # IMPORTANT: If there was any incompatibility, do NOT load optimizer and scheduler state.
        if not incompatible_keys and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if is_main_process:
                    print("   Optimizer and scheduler states loaded successfully.")
            except ValueError as e:
                if is_main_process:
                    print(f"⚠️ WARNING: Could not load optimizer state, possibly due to parameter shape mismatch. Starting with a fresh optimizer. Error: {e}")
        elif incompatible_keys:
            if is_main_process:
                print("   Skipping optimizer and scheduler state loading due to model architecture mismatch.")

        # Override learning rate if specified
        if args.override_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.override_lr
            if is_main_process:
                print(f"   Learning rate overridden to {args.override_lr}")

        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if is_main_process:
            print(f"   Resuming training from epoch {start_epoch}.")

    # Training Loop
    if is_main_process:
        print("Starting training...")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        clip_model.train()
        clip_dataset.train()  # Enable augmentation for training
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", disable=not is_main_process)

        optimizer.zero_grad(set_to_none=True)

        # Track caching stats for the epoch
        epoch_needs_inference = 0
        epoch_used_cache = 0

        for step, batch in enumerate(train_progress_bar, start=1):
            sensor_data = batch["sensor_data"].to(device, non_blocking=True)
            vlm_image_features = batch["vlm_image_features"]
            vlm_guidance_vector = batch["vlm_guidance_vector"]

            # Track caching stats
            cache_stats = batch.get("cache_stats", {})
            batch_needs_inference = cache_stats.get("needs_inference", 0)
            batch_used_cache = cache_stats.get("used_cache", 0)
            epoch_needs_inference += batch_needs_inference
            epoch_used_cache += batch_used_cache

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Forward returns scale value (exp of logit_scale) to avoid DDP issues
                s_loc, vj_loc, vi_loc, vt_loc, scale = clip_model(sensor_data, vlm_image_features, vlm_guidance_vector)

                # ===== 멀티 GPU negatives 확장 =====
                s_all  = gather_with_grad(s_loc)
                vj_all = gather_with_grad(vj_loc)
                vi_all = gather_with_grad(vi_loc)
                vt_all = gather_with_grad(vt_loc)

                total_loss, (loss_main, loss_img, loss_txt) = tri_clip_loss(
                    s_all, vj_all, vi_all, vt_all,
                    scale,
                    w_joint=1.0, w_img=0.5, w_txt=0.5
                )

            (total_loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            if is_main_process:
                # Calculate cache hit rate
                total_samples = batch_needs_inference + batch_used_cache
                cache_hit_rate = (batch_used_cache / total_samples * 100) if total_samples > 0 else 0

                train_progress_bar.set_postfix(
                    loss=total_loss.item(),
                    Lm=loss_main.item(),
                    Li=loss_img.item(),
                    Lt=loss_txt.item(),
                    lr=optimizer.param_groups[0]['lr'],
                    cache=f"{cache_hit_rate:.0f}%"
                )

        # Print epoch caching summary
        if is_main_process:
            total_epoch_samples = epoch_needs_inference + epoch_used_cache
            epoch_cache_rate = (epoch_used_cache / total_epoch_samples * 100) if total_epoch_samples > 0 else 0
            print(f"\n📊 Epoch {epoch + 1} Cache Stats: {epoch_used_cache}/{total_epoch_samples} cached ({epoch_cache_rate:.1f}%), {epoch_needs_inference} on-the-fly")


        # Validation Loop
        clip_model.eval()
        clip_dataset.eval()  # Disable augmentation for validation
        total_val_loss = 0
        val_count = 0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", disable=not is_main_process)
            for batch in val_progress_bar:
                sensor_data = batch["sensor_data"].to(device, non_blocking=True)
                vlm_image_features = batch["vlm_image_features"]  # (B, N_tokens, D_img)
                vlm_guidance_vector = batch["vlm_guidance_vector"]  # (B, D_text)
                batch_size = sensor_data.shape[0]

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # Forward returns scale value (exp of logit_scale) to avoid DDP issues
                    s_loc, vj_loc, vi_loc, vt_loc, scale = clip_model(sensor_data, vlm_image_features, vlm_guidance_vector)

                    s_all  = _all_gather_no_grad(s_loc)
                    vj_all = _all_gather_no_grad(vj_loc)
                    vi_all = _all_gather_no_grad(vi_loc)
                    vt_all = _all_gather_no_grad(vt_loc)

                    val_total, (val_main, val_img, val_txt) = tri_clip_loss(
                        s_all, vj_all, vi_all, vt_all,
                        scale,
                        w_joint=1.0, w_img=0.5, w_txt=0.5
                    )

                total_val_loss += val_total.item() * batch_size
                val_count += batch_size
                val_progress_bar.set_postfix(loss=val_total.item(), Lm=val_main.item(), Li=val_img.item(), Lt=val_txt.item())

        # gather sums across ranks
        if dist.is_initialized():
            t = torch.tensor([total_val_loss, val_count], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_val_loss, val_count = t.tolist()
        avg_val_loss = total_val_loss / max(1.0, val_count)

        
        if is_main_process:
            print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

            # Save Checkpoints
            latest_checkpoint_path = os.path.join(args.checkpoint_dir, "sensor_clip_latest.pth")
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "sensor_clip_best.pth")

            # Always save latest checkpoint
            latest_checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': clip_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_val_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }
            torch.save(latest_checkpoint_data, latest_checkpoint_path)
            print(f"💾 Latest checkpoint saved (epoch {epoch + 1}, val_loss: {avg_val_loss:.4f})")

            # Save best checkpoint if improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': clip_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': avg_val_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_val_loss
                }
                torch.save(best_checkpoint_data, best_checkpoint_path)
                print(f"✨ New best model saved with validation loss: {avg_val_loss:.4f}")

    if is_main_process: print("Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        # Set start method to 'spawn' to avoid CUDA initialization issues in forked subprocesses
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Pre-train Sensor Encoder with CLIP-style loss.")

    # Dataset & Dataloader
    parser.add_argument('--new_dataset_paths', type=str, nargs='*',
                        default=["/home/najo/NAS/VLA/dataset/New_dataset", "/home/najo/NAS/VLA/dataset/New_dataset2"],
                        help='Paths to the new format dataset directories.')
    parser.add_argument('--old_dataset_patterns', type=str, nargs='*', default=[])
    parser.add_argument('--new_weight', type=float, default=3.0, help='Weight for new datasets in weighted sampling.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers per GPU.')
    parser.add_argument('--val_split', type=float, default=0.05, help='Proportion of the dataset to use for validation.')
    parser.add_argument('--annotation_path', type=str, default="vlm_annotations.json", help='Path to the VLM annotations file.')
    parser.add_argument('--cache_root', type=str, default="/home/najo/NAS/VLA/dataset/cache", help='Root directory for all caches.')
    parser.add_argument('--disable_auto_cache_build', action='store_true',
                        help='(Deprecated) Training never auto-builds cache; uncached samples are skipped.')

    # Model & Architecture
    parser.add_argument('--vlm_model', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help='VLM model for cache building (used if cache needs to be generated).')
    parser.add_argument('--sensor_window_size', type=int, default=60, help='Temporal window size for sensor data.')
    parser.add_argument('--sensor_output_dim', type=int, default=1024, help='Output dimension of the sensor encoder.')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the shared embedding space.')
    parser.add_argument('--intermediate_vlm_dim', type=int, default=2048,
                        help='Intermediate dimension for VLM features to reduce model size. Default 2048 targets ~500MB.')
    parser.add_argument('--sensor_precision', type=str, choices=['fp32', 'bf16'], default='fp32',
                        help='Precision used for the sensor encoder. fp32 avoids cuDNN plan failures on some GPUs.')
    parser.add_argument('--skip_cache_verification', action='store_true',
                        help='Skip verifying the existence of cached VLM features during dataset initialization for faster startup. Use with caution.')
    parser.add_argument('--cache_only_mode', action='store_true',
                        help='Run in cache-only mode. VLM model will not be loaded. Fails if any cache is missing.')

    # Training & Optimization
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--grad_accum', type=int, default=1, help='Number of gradient accumulation steps.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine scheduler.')
    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='Enable find_unused_parameters=True in DDP (defaults to False for better performance).')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='/home/najo/NAS/VLA/Insertion_VLAv2/checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint to resume training from.')
    parser.add_argument('--override_lr', type=float, default=None, help='Override learning rate when resuming from checkpoint.')

    # Dataset loading optimization
    parser.add_argument('--skip_dataset_stats', action='store_true', help='Skip dataset statistics collection for faster startup')
    parser.add_argument('--force_on_the_fly', action='store_true', help='Force on-the-fly VLM inference, ignoring existing cache.')

    args = parser.parse_args()
    main(args)
