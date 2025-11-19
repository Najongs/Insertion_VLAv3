"""
Handles Vision-Language (VL) feature encoding using a Qwen-VL model.

This module provides the VisionLanguageEncoder class, which encapsulates the logic
for encoding text and image inputs into feature embeddings. It supports:
- Parallel and sequential encoding for multi-view images.
- A file-based internal caching system to speed up repeated encodings.
- Aggregation of features from multiple image views.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .vl_cache import VLACacheManager


class VisionLanguageEncoder(torch.nn.Module):
    """
    Encapsulates the logic for encoding vision-language features using a Qwen-VL model.

    This class manages the complexities of VL feature extraction, including handling
    multi-view images (sequentially or in parallel), managing an internal cache to
    avoid re-computation, and placing tensors on the correct device.

    Args:
        vl_model (Qwen2_5_VLForConditionalGeneration): The pre-loaded Qwen-VL model.
        processor (AutoProcessor): The processor corresponding to the VL model.
        cache_dir (str): Directory to store internal cache files.
        parallel_view_encoding (bool): Whether to encode multi-view images in parallel.
        view_aggregation (str): Method to aggregate features from multiple views ('mean', 'max').
        device (torch.device): The device to run the encoding on.
    """

    def __init__(
        self,
        vl_model: Qwen2_5_VLForConditionalGeneration,
        processor: AutoProcessor,
        cache_dir: str,
        parallel_view_encoding: bool,
        device: torch.device,
        view_aggregation: str = 'weighted_mean',
        view5_weight: float = 2.0,
        debug_logging: bool = False,
    ):
        super().__init__()
        self.vl_model = vl_model
        self.processor = processor
        self.cache_dir = Path(cache_dir)
        self.parallel_view_encoding = parallel_view_encoding
        self.view_aggregation = view_aggregation
        self.device = device
        self.view5_weight = view5_weight
        self.debug_logging = bool(debug_logging)
        # Qwen-VL의 ViT-G는 16x16 패치 크기에 448x448 해상도를 사용하므로, (448/16)^2 = 28^2 = 784개의 패치 토큰을 가집니다.
        self.num_patches_per_image = 784

        # Default cache settings
        self.cache_enabled = True
        self.cache_limit_gb = 20.0
        self.strict_cache = False

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def set_cache_enabled(self, enabled: bool = True):
        """Enable or disable the internal cache."""
        self.cache_enabled = enabled

    def set_strict_cache(self, enabled: bool = True):
        """
        Set strict mode for caching. If True, raises FileNotFoundError for cache misses.
        """
        self.strict_cache = enabled

    def set_cache_limit_gb(self, limit_gb: float):
        """Set the maximum size of the cache in gigabytes."""
        self.cache_limit_gb = float(limit_gb)

    def encode(
        self,
        text_inputs: List[str],
        image_inputs: List[List[str]],
        cache_keys: List[str],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main method to encode VL features. Routes to parallel or sequential implementation.

        Args:
            text_inputs (List[str]): A list of text prompts.
            image_inputs (List[List[str]]): A list of lists of image paths.
            cache_keys (List[str]): A list of unique keys for caching.
            use_cache (bool): Whether to use the cache for this encoding operation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - image_features (torch.Tensor): Aggregated (and weighted) image patch tokens.
                - text_features (torch.Tensor): Pooled text tokens for guidance.
        """
        # Parallel encoding is faster but doesn't support caching, so only use it when cache is off.
        if self.parallel_view_encoding and not use_cache:
            if not hasattr(self, "_parallel_encoding_confirmed"):
                print("🚀 VisionLanguageEncoder: Using parallel view encoding.")
                self._parallel_encoding_confirmed = True
            return self._encode_parallel(text_inputs, image_inputs)
        else:
            if not hasattr(self, "_sequential_encoding_confirmed"):
                print("📝 VisionLanguageEncoder: Using sequential view encoding with caching.")
                self._sequential_encoding_confirmed = True
            return self._encode_sequential(
                text_inputs, image_inputs, cache_keys, use_cache
            )

    # ✅ _get_cache_path()와 _enforce_cache_limit() 제거됨
    # 캐싱은 unified_model의 auto_cache_backfill이 VLACacheManager로 처리

    def _preprocess_single_input(
        self, args: Tuple[str, List[str], str]
    ) -> Tuple[str, str, List[str], str, List, List]:
        """Preprocess a single text-image pair for the Qwen-VL model."""
        txt, views, key = args

        # DEBUG: Check input
        if self.debug_logging and not hasattr(self, "_preprocess_debug_logged"):
            print(f"🔍 PREPROCESS DEBUG:")
            print(f"   txt: {txt[:50]}..." if txt else "   txt: None")
            print(f"   views: {views}")
            print(f"   views type: {type(views)}")
            if views:
                print(f"   views length: {len(views)}")
                print(f"   first view: {views[0] if len(views) > 0 else 'N/A'}")

        msg_content = [
            {"type": "image", "image": v} for v in (views or []) if v is not None
        ]

        if self.debug_logging and not hasattr(self, "_preprocess_debug_logged"):
            print(f"   msg_content images: {len([m for m in msg_content if m['type'] == 'image'])}")
            self._preprocess_debug_logged = True

        msg_content.append({"type": "text", "text": txt})
        messages = [{"role": "user", "content": msg_content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        vision_inputs, video_inputs = process_vision_info(messages)

        if self.debug_logging and not hasattr(self, "_vision_process_logged"):
            print(f"   process_vision_info returned: {vision_inputs}")
            self._vision_process_logged = True

        return key, txt, views, text, vision_inputs, video_inputs

    def _encode_sequential(
        self,
        text_inputs: List[str],
        image_inputs: List[List[str]],
        cache_keys: List[str],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순차적으로 특징을 인코딩하며, 캐시 읽기/쓰기를 지원합니다.
        V2: 이미지 특징 시퀀스와 텍스트 가이던스 벡터를 분리하여 반환합니다.
        (v6 "No Text" Plan)
        """
        features_dict = {}
        miss_items = []

        text_inputs = text_inputs or []
        image_inputs = image_inputs or [[] for _ in range(len(text_inputs))]
        cache_keys = cache_keys or [f"seq_{i}" for i in range(len(text_inputs))]

        n = min(len(text_inputs), len(image_inputs), len(cache_keys))
        text_inputs, image_inputs, cache_keys = text_inputs[:n], image_inputs[:n], cache_keys[:n]

        # ✅ vl_encoder는 캐시 로드하지 않음 - unified_model이 외부 캐시 관리 담당
        # 모든 항목을 VLM forward로 처리
        miss_items = list(zip(text_inputs, image_inputs, cache_keys))

        if miss_items and use_cache and self.strict_cache:
            missing_keys = [key for _, _, key in miss_items]
            raise FileNotFoundError(f"Strict cache mode: 다음 키에 대한 캐시 특징 없음: {missing_keys}")

        if miss_items:
            with ThreadPoolExecutor(max_workers=24) as executor:
                preprocessed_args = [(item[0], item[1], item[2]) for item in miss_items]
                results_iter = executor.map(self._preprocess_single_input, preprocessed_args)
                results = list(results_iter)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for key, txt, views, _, vision_inputs, _ in results:
                    # DEBUG: Check vision_inputs
                    if self.debug_logging and not hasattr(self, "_vision_debug_logged"):
                        print(f"🔍 VL_ENCODER DEBUG:")
                        print(f"   views: {views}")
                        print(f"   vision_inputs: {vision_inputs}")
                        print(f"   vision_inputs type: {type(vision_inputs)}")
                        if vision_inputs:
                            print(f"   vision_inputs length: {len(vision_inputs)}")
                        self._vision_debug_logged = True

                    # 1. 이미지 전용 추론 (순수 이미지 특징 추출)
                    if vision_inputs:
                        # v7: 플레이스홀더가 포함된 올바른 프롬프트 생성
                        msg_content = [{"type": "image", "image": v} for v in vision_inputs]
                        msg_content.append({"type": "text", "text": ""}) # 빈 텍스트
                        messages = [{"role": "user", "content": msg_content}]
                        text_with_placeholders = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )

                        image_only_inputs_cpu = self.processor(
                            text=[text_with_placeholders], images=vision_inputs, padding=True, return_tensors="pt"
                        )
                        image_only_inputs = {
                            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                            for k, v in image_only_inputs_cpu.items()
                        }
                        image_only_inputs['input_ids'] = image_only_inputs['input_ids'].to(dtype=torch.long)

                        # DEBUG: Check input_ids and find correct image token
                        if self.debug_logging and not hasattr(self, "_token_debug_logged"):
                            print(f"🔍 TOKEN DEBUG:")
                            print(f"   input_ids shape: {image_only_inputs['input_ids'].shape}")
                            print(f"   Unique token IDs: {torch.unique(image_only_inputs['input_ids'])}")

                            # Check special tokens in processor
                            print(f"\n📝 Processor special tokens:")
                            if hasattr(self.processor, 'tokenizer'):
                                tok = self.processor.tokenizer
                                print(f"   image_token_id: {getattr(tok, 'image_token_id', 'N/A')}")
                                print(f"   vision_start_token_id: {getattr(tok, 'vision_start_token_id', 'N/A')}")
                                print(f"   vision_end_token_id: {getattr(tok, 'vision_end_token_id', 'N/A')}")

                                # Check added_tokens
                                if hasattr(tok, 'added_tokens_decoder'):
                                    print(f"\n   Special tokens in vocab:")
                                    for token_id, token_obj in tok.added_tokens_decoder.items():
                                        token_str = str(token_obj)
                                        if 'image' in token_str.lower() or 'vision' in token_str.lower():
                                            print(f"      {token_id}: {token_str}")

                            if self.debug_logging:
                                print(f"\n   text_with_placeholders sample:")
                                print(f"   {text_with_placeholders[:500]}")

                            self._token_debug_logged = True

                        image_outputs = self.vl_model(**image_only_inputs, output_hidden_states=True, return_dict=True)
                        image_hidden_state = image_outputs.hidden_states[-1]

                        # Qwen2.5-VL uses <|vision_start|> (151652) and <|vision_end|> (151653)
                        # Extract all hidden states between these markers
                        input_ids_flat = image_only_inputs['input_ids'].squeeze(0)

                        # Find all vision_start and vision_end positions
                        vision_start_positions = torch.where(input_ids_flat == 151652)[0]
                        vision_end_positions = torch.where(input_ids_flat == 151653)[0]

                        if self.debug_logging and not hasattr(self, "_indices_debug_logged"):
                            print(f"\n🔍 VISION TOKEN EXTRACTION:")
                            print(f"   vision_start_positions: {vision_start_positions}")
                            print(f"   vision_end_positions: {vision_end_positions}")
                            self._indices_debug_logged = True

                        # Collect all image patch tokens between vision_start and vision_end
                        image_patch_indices = []
                        for start_pos, end_pos in zip(vision_start_positions, vision_end_positions):
                            # Include all tokens between start and end (exclusive of markers)
                            patch_indices = torch.arange(start_pos + 1, end_pos, device=input_ids_flat.device)
                            image_patch_indices.append(patch_indices)

                        if image_patch_indices:
                            image_patch_indices = torch.cat(image_patch_indices)
                            image_features = image_hidden_state[:, image_patch_indices, :]

                            if self.debug_logging and not hasattr(self, "_extraction_debug_logged"):
                                print(f"   Total image patches extracted: {len(image_patch_indices)}")
                                print(f"   image_features shape: {image_features.shape}")
                                self._extraction_debug_logged = True
                        else:
                            image_features = torch.empty(1, 0, self.vl_model.config.hidden_size, device=self.device, dtype=torch.bfloat16)
                    else:
                        image_features = torch.empty(1, 0, self.vl_model.config.hidden_size, device=self.device, dtype=torch.bfloat16)

                    # 2. 텍스트 전용 추론 (가이던스 벡터 추출)
                    if txt:
                        text_only_inputs_cpu = self.processor(
                            text=[txt], images=None, padding=True, return_tensors="pt"
                        )
                        text_only_inputs = {
                            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                            for k, v in text_only_inputs_cpu.items()
                        }
                        text_only_inputs['input_ids'] = text_only_inputs['input_ids'].to(dtype=torch.long)

                        text_outputs = self.vl_model(**text_only_inputs, output_hidden_states=True, return_dict=True)
                        text_hidden_state = text_outputs.hidden_states[-1]
                        guidance_vector = text_hidden_state.mean(dim=1)
                    else:
                        guidance_vector = torch.zeros(1, self.vl_model.config.hidden_size, device=self.device, dtype=torch.bfloat16)

                    # View5 가중치 적용 (필요 시)
                    if views and self.view_aggregation == 'weighted_mean' and image_features.numel() > 0:
                        num_patches_per_view = image_features.shape[1] // len(views)
                        weights = torch.ones(len(views), device=self.device, dtype=torch.bfloat16)
                        for i, view_path in enumerate(views):
                            if 'View5' in view_path:
                                weights[i] = self.view5_weight
                        
                        weights_expanded = weights.repeat_interleave(num_patches_per_view).unsqueeze(0).unsqueeze(-1)
                        image_features = image_features * weights_expanded

                    # ✅ 캐싱은 unified_model의 auto_cache_backfill이 VLACacheManager로 처리
                    # vl_encoder는 순수하게 VLM forward만 수행
                    features_dict[key] = (image_features, guidance_vector)

        if not features_dict:
            raise RuntimeError("유효한 VL 토큰을 로드하거나 생성할 수 없습니다. 입력을 확인하세요.")

        final_features_list = [features_dict[key] for key in cache_keys if key in features_dict]
        if len(final_features_list) != len(cache_keys):
            raise RuntimeError("내부 캐시 처리 실패. 토큰 수가 일치하지 않습니다.")

        final_image_features = torch.cat([item[0] for item in final_features_list], dim=0)
        final_guidance_vectors = torch.cat([item[1] for item in final_features_list], dim=0)

        return final_image_features, final_guidance_vectors

    def _encode_parallel(
        self, text_inputs: List[str], image_inputs: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        병렬로 특징을 인코딩하며, 한 번에 여러 뷰를 처리합니다.
        V2: 이미지 특징 시퀀스와 텍스트 가이던스 벡터를 분리하여 반환합니다.
        (v6 "No Text" Plan)
        """
        batch_features_list = []

        text_inputs = text_inputs or []
        image_inputs = image_inputs or [[] for _ in range(len(text_inputs))]

        n = min(len(text_inputs), len(image_inputs))
        text_inputs, image_inputs = text_inputs[:n], image_inputs[:n]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for txt, views in zip(text_inputs, image_inputs):
                # 1. 이미지 전용 추론
                if views:
                    # v7: 플레이스홀더가 포함된 올바른 프롬프트 생성
                    texts_with_placeholders = []
                    for _ in views:
                        msg_content = [{"type": "image"}, {"type": "text", "text": ""}]
                        messages = [{"role": "user", "content": msg_content}]
                        text_with_placeholders = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                        texts_with_placeholders.append(text_with_placeholders)

                    image_only_inputs_cpu = self.processor(
                        text=texts_with_placeholders, images=views, padding=True, return_tensors="pt"
                    )
                    image_only_inputs = {
                        k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in image_only_inputs_cpu.items()
                    }
                    image_only_inputs['input_ids'] = image_only_inputs['input_ids'].to(dtype=torch.long)

                    image_outputs = self.vl_model(**image_only_inputs, output_hidden_states=True, return_dict=True)
                    image_hidden_state = image_outputs.hidden_states[-1]
                    
                    image_token_mask = (image_only_inputs['input_ids'] == 151857)
                    
                    image_features_per_view = []
                    for i in range(len(views)):
                        view_indices = torch.where(image_token_mask[i])[0]
                        image_features_per_view.append(image_hidden_state[i, view_indices, :])
                    
                    if self.view_aggregation == 'weighted_mean':
                        weights = torch.ones(len(views), device=self.device, dtype=torch.bfloat16)
                        for i, view_path in enumerate(views):
                            # Handle both path strings and PIL Images for testing
                            if isinstance(view_path, str) and 'View5' in view_path:
                                weights[i] = self.view5_weight
                        
                        for i in range(len(views)):
                            image_features_per_view[i] = image_features_per_view[i] * weights[i]
                            
                    aggregated_image_features = torch.cat(image_features_per_view, dim=0).unsqueeze(0)
                else:
                    aggregated_image_features = torch.empty(1, 0, self.vl_model.config.hidden_size, device=self.device, dtype=torch.bfloat16)

                # 2. 텍스트 전용 추론
                if txt:
                    text_only_inputs_cpu = self.processor(
                        text=[txt], images=None, padding=True, return_tensors="pt"
                    )
                    text_only_inputs = {
                        k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in text_only_inputs_cpu.items()
                    }
                    text_only_inputs['input_ids'] = text_only_inputs['input_ids'].to(dtype=torch.long)

                    text_outputs = self.vl_model(**text_only_inputs, output_hidden_states=True, return_dict=True)
                    text_hidden_state = text_outputs.hidden_states[-1]
                    guidance_vector = text_hidden_state.mean(dim=1)
                else:
                    guidance_vector = torch.zeros(1, self.vl_model.config.hidden_size, device=self.device, dtype=torch.bfloat16)

                batch_features_list.append((aggregated_image_features, guidance_vector))

        if not batch_features_list:
            raise RuntimeError("VL 특징을 인코딩할 수 없습니다. 입력을 확인하세요.")

        final_image_features = torch.cat([item[0] for item in batch_features_list], dim=0)
        final_guidance_vectors = torch.cat([item[1] for item in batch_features_list], dim=0)

        return final_image_features, final_guidance_vectors
