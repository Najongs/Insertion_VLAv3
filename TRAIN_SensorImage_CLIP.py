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

CLIP_PROMPT_TEXT = (
    "This is a robot hand-eye view with a sensorized needle. Focus on the needle tip and its "
    "interaction with the environment. Analyze the following aspects:\\n"
    "1. Proximity: How close is the needle tip to the intended target and other nearby objects? "
    "(e.g., far, near, touching).\\n"
    "2. Contact State: Is the needle tip making contact with any surface? Describe the nature of the "
    "contact (e.g., no contact, light touch, firm press, inserting).\\n"
    "3. Certainty: How certain are you about the contact state? (High, Medium, Low)."
)


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


def get_clip_prompt_hash() -> str:
    """Shared helper so cache building and training use the exact same prompt hash."""
    return hashlib.md5(CLIP_PROMPT_TEXT.encode()).hexdigest()[:8]


def generate_text_response(
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
            image = sample.get("hand_eye_image")
            episode_id = sample.get("episode_id")
            vlm_idx = sample.get("vlm_idx")

            if vlm_idx is None or episode_id is None or image is None:
                skipped_count += 1
                continue

            # Check if cache already exists
            if cache_manager.cache_exists(dataset_name=episode_id, vlm_idx=vlm_idx, prompt_hash=prompt_hash):
                skipped_count += 1
                continue

            # Generate features
            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": CLIP_PROMPT_TEXT}
                    ]
                }]
                generation_text_input = vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                vision_input, _ = process_vision_info(messages)

                text_response = generate_text_response(
                    vlm_model, vlm_processor, generation_text_input, vision_input, max_new_tokens
                )

                with torch.no_grad():
                    # 1. Image-only inference (extract all image tokens)
                    image_only_messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": vision_input},
                            {"type": "text", "text": ""}
                        ]
                    }]
                    image_text_with_placeholders = vlm_processor.apply_chat_template(
                        image_only_messages, tokenize=False, add_generation_prompt=False
                    )
                    image_inputs = vlm_processor(
                        text=[image_text_with_placeholders],
                        images=[vision_input],
                        padding=True,
                        return_tensors="pt"
                    ).to(device=vlm_model.device, dtype=vlm_model.dtype)

                    image_outputs = vlm_model(**image_inputs, output_hidden_states=True, return_dict=True)
                    image_hidden_state = image_outputs.hidden_states[-1]

                    # Extract image tokens (token ID 151857)
                    image_token_mask = (image_inputs['input_ids'] == 151857)
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
                    prompt_hash=prompt_hash,
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
    All augmentations have ≤30% probability.
    """
    def __init__(self,
                 time_mask_ratio=0.1,
                 noise_std=0.005,
                 scale_range=(0.97, 1.03)):
        self.time_mask_ratio = time_mask_ratio
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, sensor_data):
        """
        Args:
            sensor_data: (T, C=1026) - distance features (1-1025) + force (1026)
        """
        augmented = sensor_data.clone()
        device = augmented.device

        # 1. Time masking (20% probability)
        if np.random.random() < 0.20:
            T = augmented.shape[0]
            num_mask = int(T * self.time_mask_ratio)
            if num_mask > 0:
                mask_indices = torch.randperm(T, device=device)[:num_mask]
                augmented[mask_indices] = 0.0

        # 2. Gaussian noise (25% probability)
        if np.random.random() < 0.25:
            noise = torch.randn_like(augmented, device=device) * self.noise_std
            # Force channel (last) gets slightly more noise
            noise[:, -1] *= 1.5
            augmented += noise

        # 3. Magnitude scaling (30% probability)
        if np.random.random() < 0.30:
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
                 mode: Literal["train", "cache_build"] = "train"):
        self.unified_dataset = unified_dataset
        self.hand_eye_view_keyword = "View5" # Or "Oak"
        self.vlm_annotations = vlm_annotations if vlm_annotations is not None else {}
        self.cache_path = cache_path # Cache for filtered indices
        self.mode = mode
        if self.mode not in ("train", "cache_build"):
            raise ValueError(f"Unsupported mode '{self.mode}'. Expected 'train' or 'cache_build'.")
        self.require_cached_features = self.mode != "cache_build"
        self._vlm_idx_cache: Dict[Tuple[str, int], Optional[int]] = {}

        # Cache for VLM features
        self.clip_cache_manager = None
        self.clip_prompt_hash = None
        if self.require_cached_features and not clip_cache_root:
            raise ValueError("clip_cache_root must be provided when mode!='cache_build'")
        if clip_cache_root and self.require_cached_features:
            self.clip_cache_manager = VLACacheManager(cache_dir=clip_cache_root)
            self.clip_prompt_hash = get_clip_prompt_hash()
            print(f"   ... Using CLIP VLM feature cache at: {clip_cache_root}")
            print(f"   ... Using prompt_hash: {self.clip_prompt_hash}")

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
            pbar_datasets.set_postfix(episode=episode_id)

            # Per-episode cache file
            episode_cache_path = cache_dir / f"{episode_id}.json"

            # Try to load from cache
            episode_valid_indices = None
            cached_data = _load_cached_indices(cache_dir, episode_id)
            if cached_data is None and fallback_cache_dir != cache_dir:
                cached_data = _load_cached_indices(fallback_cache_dir, episode_id)

            if cached_data is not None:
                cached_prompt_hash = cached_data.get("prompt_hash")
                if self.require_cached_features and cached_prompt_hash != self.clip_prompt_hash:
                    episode_valid_indices = None
                else:
                    episode_valid_indices = cached_data["indices"]
                    episode_valid_indices = self._ensure_cached_features_exist(
                        episode_valid_indices, sub_dataset, episode_id, is_main_process=is_main_process
                    )
                    cached_data["indices"] = episode_valid_indices
                    cached_data["mode"] = self.mode
                    cached_data["prompt_hash"] = self.clip_prompt_hash
                    if is_main_process:
                        with open(episode_cache_path, 'w') as f:
                            json.dump(cached_data, f, indent=2)
                    if dist.is_initialized():
                        dist.barrier()

            # If cache miss, filter this episode
            if episode_valid_indices is None:
                total_episodes += 1

                # Strategy: Use last 20% of episode data (80% onwards)
                # BUT only include samples that have CLIP VLM cache
                start_idx = int(num_samples_in_episode * 0.8)
                candidate_indices = list(range(start_idx, num_samples_in_episode))

                episode_valid_indices = []
                if self.require_cached_features:
                    if not self.clip_cache_manager or not self.clip_prompt_hash:
                        raise RuntimeError("CLIP cache manager not initialized but required for filtering.")
                    for sample_idx_local in candidate_indices:
                        vlm_idx = self._get_vlm_idx_for_sample(sub_dataset, episode_id, sample_idx_local)
                        if vlm_idx is None:
                            continue

                        # Check if cache exists for this sample
                        if self.clip_cache_manager.cache_exists(
                            dataset_name=episode_id,
                            vlm_idx=vlm_idx,
                            prompt_hash=self.clip_prompt_hash
                        ):
                            episode_valid_indices.append(sample_idx_local)
                else:
                    episode_valid_indices = candidate_indices

                # Save to cache (main process only)
                if is_main_process:
                    with open(episode_cache_path, 'w') as f:
                        json.dump({
                            "indices": episode_valid_indices,
                            "episode_id": episode_id,
                            "start_idx": start_idx,
                            "total_samples": num_samples_in_episode,
                            "candidate_samples": len(candidate_indices),
                            "valid_samples": len(episode_valid_indices),
                            "strategy": "last_20_percent_with_cache" if self.require_cached_features else "last_20_percent_all",
                            "mode": self.mode,
                            "prompt_hash": self.clip_prompt_hash,
                        }, f)

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
    ) -> List[int]:
        """
        Revalidate cached indices loaded from disk and drop samples whose VLM cache file
        has been removed since the cache file was first created.
        """
        if (
            not episode_valid_indices
            or not self.require_cached_features
            or self.clip_cache_manager is None
            or not self.clip_prompt_hash
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
                prompt_hash=self.clip_prompt_hash,
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

    def _load_hand_eye_image(self, sample):
        """
        Select and load the hand-eye camera image for cache building mode.
        Prioritizes paths containing the hand-eye keyword, falls back to the first available view.
        """
        # Try to get from sample["images"] first
        image_entries = sample.get("images") or []
        selected_entry = None

        def matches_keyword(entry_path: str) -> bool:
            return self.hand_eye_view_keyword.lower() in entry_path.lower()

        for entry in image_entries:
            if entry is None:
                continue
            if isinstance(entry, Image.Image):
                if selected_entry is None:
                    selected_entry = entry
                continue

            entry_path = str(entry)
            if not entry_path:
                continue
            if matches_keyword(entry_path):
                selected_entry = entry_path
                break
            if selected_entry is None:
                selected_entry = entry_path

        # If no image found from sample["images"], try to construct path manually
        if selected_entry is None:
            episode_id = sample.get("episode_id")
            vlm_idx = sample.get("vlm_idx")

            if episode_id and vlm_idx is not None:
                # Find the dataset containing this episode
                for sub_dataset in self.unified_dataset.datasets:
                    if sub_dataset.data_dir.name == episode_id:
                        # Try to find hand-eye view directory
                        data_dir = sub_dataset.data_dir

                        # Try different possible locations
                        possible_view_dirs = [
                            data_dir / self.hand_eye_view_keyword,  # New format: data_collection_*/View5
                            data_dir / "images" / self.hand_eye_view_keyword,  # Old format: episode_*/images/View5
                        ]

                        for view_dir in possible_view_dirs:
                            if view_dir.exists():
                                # Get all images in this directory
                                image_files = sorted(list(view_dir.glob("*.jpg")) + list(view_dir.glob("*.png")))
                                if image_files and vlm_idx < len(image_files):
                                    selected_entry = str(image_files[vlm_idx])
                                    break

                        if selected_entry:
                            break

        if selected_entry is None:
            raise RuntimeError(
                f"No valid image paths found for hand-eye camera view. "
                f"episode_id={sample.get('episode_id')}, vlm_idx={sample.get('vlm_idx')}"
            )

        if isinstance(selected_entry, Image.Image):
            return selected_entry

        return Image.open(selected_entry).convert("RGB")

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
            # Load dictionary of features from CLIP VLM feature cache
            if not self.clip_cache_manager or episode_id is None or vlm_idx is None or not self.clip_prompt_hash:
                raise RuntimeError(
                    f"CLIP cache manager not initialized or missing metadata. "
                    f"episode_id={episode_id}, vlm_idx={vlm_idx}, prompt_hash={self.clip_prompt_hash}"
                )
            cached_features = self._load_cached_vlm_feature(episode_id, vlm_idx)
            vlm_image_features = cached_features['image_features']  # (N_tokens, D)
            vlm_guidance_vector = cached_features['guidance_vector']  # (D,)
        else:
            # In cache_build mode, we load the raw image
            hand_eye_representation = self._load_hand_eye_image(sample)
            # Placeholders for the items that will be created from this image
            vlm_image_features = hand_eye_representation # Pass image to be processed
            vlm_guidance_vector = torch.empty(0) # Not used in cache build mode

        timestamp = float(timestamp if timestamp is not None else 0.0)
        vlm_description = self.vlm_annotations.get(episode_id, {}).get(str(timestamp), "no_vlm_description")

        # Apply sensor augmentation (only during training)
        if self.use_augmentation and self.is_training:
            sensor_data = self.sensor_aug(sensor_data)

        if self.require_cached_features:
            return {
                "sensor_data": sensor_data,
                "vlm_image_features": vlm_image_features,
                "vlm_guidance_vector": vlm_guidance_vector,
                "vlm_description": vlm_description,
                "episode_id": episode_id,
                "timestamp": timestamp,
                "vlm_idx": vlm_idx,
                "vlm_cache_key": vlm_cache_key,
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

    def _load_cached_vlm_feature(self, episode_id: str, vlm_idx: int) -> Dict[str, torch.Tensor]:
        vlm_feature = self.clip_cache_manager.load_cache(
            dataset_name=episode_id,
            vlm_idx=vlm_idx,
            prompt_hash=self.clip_prompt_hash
        )

        if vlm_feature is None:
            raise RuntimeError(
                f"CLIP VLM feature cache not found for episode={episode_id}, vlm_idx={vlm_idx}. "
                "This sample should have been filtered out. Please regenerate the filter cache."
            )

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

def clip_collate_fn(batch, window_size):
    """
    Robust collate function that pads sensor data and handles the new tuple cache format.
    - In 'train' mode, it pads 'vlm_image_features' and stacks 'vlm_guidance_vector'.
    - In 'cache_build' mode, it gathers 'hand_eye_image' into a list.
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
            # image_features: (N_tokens, D)
            # guidance_vector: (D,)
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
        # Pad image feature sequences (each sample has different N_tokens)
        collated_batch["vlm_image_features"] = torch.nn.utils.rnn.pad_sequence(
            vlm_image_features_list, batch_first=True, padding_value=0.0
        )  # (B, max_N_tokens, D)
        collated_batch["vlm_guidance_vector"] = torch.stack(vlm_guidance_vectors)  # (B, D)
    else:
        collated_batch["hand_eye_image"] = hand_eye_images

    return collated_batch


# =====================================
# 2. CLIP Model Definition
# =====================================

class CLIPModel(nn.Module):
    """
    A lightweight model that holds the sensor encoder and performs cross-attention
    to fuse VLM image features and text guidance vector.

    New architecture:
    1. Fuse image_features + guidance_vector using cross-attention
    2. Sensor encoder produces sensor embedding
    3. Contrastive learning between sensor and fused VLM embedding
    """
    def __init__(
        self,
        sensor_encoder,
        sensor_output_dim: int,
        image_embedding_dim: int,
        text_embedding_dim: int,
        projection_dim: int = 512,
    ):
        super().__init__()
        self.sensor_encoder = sensor_encoder

        # Cross-attention for VLM fusion (guidance_vector attends to image_features)
        self.vlm_fusion_attention = nn.MultiheadAttention(
            embed_dim=text_embedding_dim,  # Query dimension (guidance_vector)
            num_heads=8,
            dropout=0.1,
            batch_first=True,
            kdim=image_embedding_dim,  # Key dimension (image_features)
            vdim=image_embedding_dim,  # Value dimension (image_features)
        )

        # Projection heads
        self.sensor_projection = nn.Linear(sensor_output_dim, projection_dim)
        self.sensor_norm = nn.LayerNorm(projection_dim)

        # VLM fusion projection (from text_dim -> projection_dim)
        self.vlm_fusion_projection = nn.Linear(text_embedding_dim, projection_dim)
        self.vlm_norm = nn.LayerNorm(projection_dim)

    def forward(self, sensor_data, vlm_image_features, vlm_guidance_vector):
        """
        Args:
            sensor_data: (B, T, C) - sensor time series
            vlm_image_features: (B, N_tokens, D_img) - image token features
            vlm_guidance_vector: (B, D_text) - text guidance vector

        Returns:
            sensor_embedding: (B, D_proj) - normalized sensor embedding
            vlm_embedding: (B, D_proj) - normalized fused VLM embedding
        """
        # 1. Encode sensor data
        sensor_features = self.sensor_encoder(sensor_data)  # (B, D_sensor)

        target_device = sensor_features.device

        # Move pre-computed embeddings to the correct device
        vlm_image_features = vlm_image_features.to(target_device)
        vlm_guidance_vector = vlm_guidance_vector.to(target_device)

        # 2. Fuse VLM features: guidance_vector attends to image_features
        # Query: guidance_vector (B, 1, D_text)
        # Key/Value: image_features (B, N_tokens, D_img)
        guidance_query = vlm_guidance_vector.unsqueeze(1)  # (B, 1, D_text)

        fused_vlm_features, _ = self.vlm_fusion_attention(
            query=guidance_query,
            key=vlm_image_features,
            value=vlm_image_features,
            need_weights=False
        )
        fused_vlm_features = fused_vlm_features.squeeze(1)  # (B, D_text)

        # 3. Project and normalize sensor embedding
        sensor_embedding = self.sensor_projection(sensor_features)
        sensor_embedding = self.sensor_norm(sensor_embedding)
        sensor_embedding = F.normalize(sensor_embedding, p=2, dim=-1)

        # 4. Project and normalize VLM embedding
        vlm_embedding = self.vlm_fusion_projection(fused_vlm_features)
        vlm_embedding = self.vlm_norm(vlm_embedding)
        vlm_embedding = F.normalize(vlm_embedding, p=2, dim=-1)

        return sensor_embedding, vlm_embedding

# =====================================
# 3. SigLIP-Style Contrastive Loss
# =====================================

def siglip_loss(sensor_embeddings, image_embeddings):
    """
    SigLIP loss uses a symmetric sigmoid cross-entropy objective between
    all sensor/image pairs instead of softmax contrastive loss.
    """
    logits = torch.matmul(sensor_embeddings, image_embeddings.T) / SIGLIP_TEMPERATURE
    batch_size = logits.shape[0]
    labels = torch.eye(batch_size, device=logits.device, dtype=logits.dtype)

    loss_sensor = F.binary_cross_entropy_with_logits(logits, labels)
    loss_image = F.binary_cross_entropy_with_logits(logits.T, labels)

    return 0.5 * (loss_sensor + loss_image)

# =====================================
# 4. Main Training Function
# =====================================

def main(args):
    # DDP Setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = local_rank == 0
    device_props = torch.cuda.get_device_properties(local_rank)
    device_total_gb = device_props.total_memory / (1024 ** 3)

    if is_main_process:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")

    clip_cache_root = Path(args.cache_root) / "clip_vlm_features"
    prompt_hash = get_clip_prompt_hash()

    # Check cache; do NOT auto-build during training. Skip samples without cache.
    cache_spec = infer_cached_feature_spec(clip_cache_root, prompt_hash, return_none_if_missing=True)

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

    # NOTE: prompt_hash_override ties UnifiedVLADataset to the CLIP prompt hash that was used
    # during cache generation. If someone edits the prompt text without rebuilding the cache,
    # coverage drops to zero because hashes no longer match. Always regenerate caches (and
    # keep this override) whenever CLIP_PROMPT_TEXT changes.
    unified_dataset = create_unified_dataloader(
        new_dataset_paths=args.new_dataset_paths,
        old_dataset_patterns=args.old_dataset_patterns,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        return_dataset=True,
        use_cache=True,  # Enable per-dataset caching in unified_dataset
        cache_root=str(clip_cache_root),  # Align unified dataset cache root with CLIP cache
        prompt_hash_override=prompt_hash,
    )

    # No auto cache build path. If cache is missing, dataset filtering below will drop uncached samples.

    # Create training dataset with cache enabled
    clip_dataset = SensorImageCLIPDataset(
        unified_dataset,
        vlm_annotations=annotations,
        cache_path=None,  # Use per-episode caching instead of global cache
        clip_cache_root=str(clip_cache_root),
        mode="train"
    )

    # Now initialize model with correct dimensions
    if is_main_process:
        print("Initializing Sensor Encoder and CLIP Model...")

    sensor_encoder = ForceAwareSensorEncoder(
        temporal_length=args.sensor_window_size,
        output_dim=args.sensor_output_dim,
        use_transformer=True,
        num_transformer_layers=2
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
    ).to(device)
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
    collate_fn_with_window = partial(clip_collate_fn, window_size=args.sensor_window_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn_with_window, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, num_workers=args.num_workers,
        collate_fn=collate_fn_with_window, pin_memory=True
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

        for batch in train_progress_bar:
            sensor_data = batch["sensor_data"].to(device, non_blocking=True)
            vlm_image_features = batch["vlm_image_features"]  # (B, N_tokens, D_img)
            vlm_guidance_vector = batch["vlm_guidance_vector"]  # (B, D_text)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                sensor_embed, vlm_embed = clip_model(sensor_data, vlm_image_features, vlm_guidance_vector)
                total_loss = siglip_loss(
                    sensor_embed, vlm_embed
                )

            total_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if is_main_process:
                train_progress_bar.set_postfix(
                    loss=total_loss.item(),
                    lr=optimizer.param_groups[0]['lr']
                )

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
                    sensor_embed, vlm_embed = clip_model(sensor_data, vlm_image_features, vlm_guidance_vector)
                    val_total = siglip_loss(
                        sensor_embed, vlm_embed
                    )

                total_val_loss += val_total.item() * batch_size
                val_count += batch_size
                val_progress_bar.set_postfix(loss=val_total.item())

        avg_val_loss = total_val_loss / val_count if val_count > 0 else 0
        
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
    parser.add_argument('--sensor_output_dim', type=int, default=3072, help='Output dimension of the sensor encoder.')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the shared embedding space.')
    parser.add_argument('--sensor_precision', type=str, choices=['fp32', 'bf16'], default='fp32',
                        help='Precision used for the sensor encoder. fp32 avoids cuDNN plan failures on some GPUs.')

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

    args = parser.parse_args()
    main(args)
