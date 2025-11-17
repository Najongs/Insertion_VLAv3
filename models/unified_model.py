"""
Unified Vision-Language-Action (VLA) Model - QwenVLAUnified

This file defines the main `QwenVLAUnified` model, which integrates various
sub-modules to perform Vision-Language-Action tasks.

The `QwenVLAUnified` class acts as the central orchestrator, combining:
- A Vision-Language (VL) backbone (from `vl_encoder`).
- Optional sensor and robot state encoders (from `Encoder_model`).
- An action prediction expert (from `action_decoder`).

It supports different action prediction paradigms like Flow Matching and Regression,
and handles the fusion of multimodal data.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

# 상대 임포트 vs 절대 임포트 처리 (직접 실행 시 절대 임포트 사용)
if __name__ == "__main__":
    # 직접 실행 시 부모 디렉토리를 sys.path에 추가
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.Encoder_model import RobotStateEncoder, UnifiedGatedSensorEncoder, force_bn_fp32_
    from models.action_decoder import FlowMatchingActionExpert, RegressionActionExpert
    from models.vl_cache import VLACacheManager, get_cache_manager
    from models.vl_encoder import VisionLanguageEncoder
else:
    # 모듈로 임포트 시 상대 임포트 사용
    from .Encoder_model import RobotStateEncoder, UnifiedGatedSensorEncoder, force_bn_fp32_
    from .action_decoder import FlowMatchingActionExpert, RegressionActionExpert
    from .vl_cache import VLACacheManager, get_cache_manager
    from .vl_encoder import VisionLanguageEncoder


class QwenVLAUnified(nn.Module):
    """
    Qwen-VL 백본을 기반으로 한 통합 Vision-Language-Action (VLA) 모델입니다.
    센서 융합 및 로봇 상태 인코더를 통해 다양한 양식의 데이터를 통합하여 행동 예측을 수행합니다.
    Flow Matching 또는 Regression 기반의 행동 전문가를 선택할 수 있습니다.
    """
    def __init__(
        self,
        model_type: Literal['regression', 'flow_matching'] = 'flow_matching',
        vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        action_dim=7,
        horizon=8,
        hidden_dim=1024,
        cache_dir="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
        external_cache_root: Optional[str] = None,
        auto_cache_backfill: bool = True,
        # --- 통합된 인코더 파라미터 ---
        sensor_enabled=True,
        sensor_input_channels=1026, # dist_channels(1025) + force_channels(1)
        sensor_temporal_length=65,
        sensor_output_dim=3072, # UnifiedGatedSensorEncoder의 기본 출력 차원
        robot_state_enabled=True,
        robot_state_temporal_length=100,
        robot_state_output_dim=1024, # 업그레이드된 RobotStateEncoder의 기본 출력 차원
        # --- 나머지 파라미터 ---
        fusion_strategy='cross_attention',
        flow_steps=10,
        flow_solver='euler',
        finetune_vl='none',
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        image_resize_height=None,
        image_resize_width=None,
        parallel_view_encoding=False,
        view_aggregation='weighted_mean',
        view5_weight=2.0,
        device_map=None,
        cache_only_mode=False):
        super().__init__()

        if model_type not in ['regression', 'flow_matching']:
            raise ValueError(f"model_type은 'regression', 'flow_matching' 중 하나여야 합니다. 현재: {model_type}")

        self.model_type = model_type
        self.sensor_enabled = sensor_enabled
        self.robot_state_enabled = robot_state_enabled
        self.fusion_strategy = fusion_strategy
        self.flow_steps = flow_steps
        self.flow_solver = flow_solver
        self.action_dim = action_dim
        self.horizon = horizon
        self.auto_cache_backfill = auto_cache_backfill
        self.cache_only_mode = cache_only_mode
        self.external_cache_mgr: Optional[VLACacheManager] = None
        self.external_cache_root = None

        if external_cache_root:
            self.external_cache_root = str(external_cache_root)
            try:
                self.external_cache_mgr = get_cache_manager(cache_dir=self.external_cache_root)
                if self.auto_cache_backfill:
                    print(f"   자동 캐시 백필 활성화 → {self.external_cache_root}")
            except Exception as e:
                print(f"⚠️ 외부 캐시 관리자 초기화 실패 ({external_cache_root}): {e}")

        print(f"🚀 QwenVLA 통합 모델 V3 (Unified Encoders) 로딩 중")
        print(f"   모델 타입: {model_type.upper()}")
        print(f"   센서 활성화: {sensor_enabled}")
        print(f"   로봇 상태 활성화: {robot_state_enabled}")
        if cache_only_mode:
            print(f"   ⚡ 캐시 전용 모드: VLM 모델 로드 스킵 (메모리 절약)")

        # VLM 로딩 (cache_only_mode가 아닐 때만)
        if not cache_only_mode:
            self.processor = AutoProcessor.from_pretrained(vl_model_name, use_fast=False)

            if image_resize_height and image_resize_width:
                target_pixels = image_resize_height * image_resize_width
                self.processor.image_processor.min_pixels = target_pixels
                self.processor.image_processor.max_pixels = target_pixels
                print(f"   이미지 리사이즈: {image_resize_width}x{image_resize_height}")

            self.vl_model = self._load_qwen_with_fallback(vl_model_name, device_map)
            print(f"   VL 모델 hidden_size: {self.vl_model.config.hidden_size}")

            self.vl_encoder = VisionLanguageEncoder(
                vl_model=self.vl_model,
                processor=self.processor,
                cache_dir=cache_dir,
                parallel_view_encoding=parallel_view_encoding,
                view_aggregation=view_aggregation,
                view5_weight=view5_weight,
                device=next(self.vl_model.parameters()).device
            )

            if finetune_vl == 'lora':
                print(f"🔧 VL 모델에 LoRA 적용 중 (r={lora_r})...")
                lora_config = LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM"
                )
                self.vl_model = get_peft_model(self.vl_model, lora_config)
                print("✅ LoRA 적용 완료.")
            elif finetune_vl == 'none':
                print("🧊 VL 모델 매개변수 동결 중...")
                for p in self.vl_model.parameters():
                    p.requires_grad = False
                print("✅ VL 모델 동결 완료.")
            else:
                print("🔥 VL 모델은 완전히 학습 가능합니다.")

            vl_hidden_size = self.vl_model.config.hidden_size
        else:
            # 캐시 전용 모드: VLM을 로드하지 않고 기본 차원만 설정
            self.processor = None
            self.vl_model = None
            self.vl_encoder = None
            inferred_dim = self._infer_cached_vl_hidden_size(external_cache_root)
            if inferred_dim:
                vl_hidden_size = inferred_dim
                print(f"   📦 캐시 기반 hidden_size 자동 감지: {vl_hidden_size}")
            else:
                vl_hidden_size = 2048  # 기본값 (Qwen2.5-VL-3B)
                print(f"   ⚠️ 캐시에서 hidden_size를 찾지 못해 기본값 {vl_hidden_size} 사용")

        if sensor_enabled:
            print("   센서 인코더: UnifiedGatedSensorEncoder (bfloat16 ~53MB)")
            self.sensor_encoder = UnifiedGatedSensorEncoder(
                dist_channels=sensor_input_channels - 1,
                force_channels=1,
                temporal_length=sensor_temporal_length,
                output_dim=sensor_output_dim
            ).to(dtype=torch.bfloat16, device="cuda")
            force_bn_fp32_(self.sensor_encoder)
        else:
            self.sensor_encoder = None

        if self.robot_state_enabled:
            print("   로봇 상태 인코더: Upgraded RobotStateEncoder (bfloat16 ~41MB)")
            self.robot_state_encoder = RobotStateEncoder(
                temporal_length=robot_state_temporal_length,
                output_dim=robot_state_output_dim
            ).to(dtype=torch.bfloat16, device="cuda")
        else:
            self.robot_state_encoder = None

        combined_sensor_dim = 0
        if sensor_enabled:
            combined_sensor_dim += sensor_output_dim
        if self.robot_state_enabled:
            combined_sensor_dim += robot_state_output_dim

        ActionExpertClass = FlowMatchingActionExpert if model_type == 'flow_matching' else RegressionActionExpert
        self.action_expert = ActionExpertClass(
            image_feature_dim=vl_hidden_size,
            text_guidance_dim=vl_hidden_size,
            sensor_dim=combined_sensor_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
        ).to(dtype=torch.bfloat16, device="cuda")

        print("✅ 모델 초기화 완료!")

    def _load_qwen_with_fallback(self, vl_model_name: str, device_map: Optional[str]) -> Qwen2_5_VLForConditionalGeneration:
        """
        Qwen-VL 모델을 로드합니다. FlashAttention 2 또는 SDPA 어텐션 구현에 대한 폴백 메커니즘이 포함됩니다.
        GPU 메모리 사용량에 따라 bfloat16 또는 float16을 시도합니다.
        """
        dtype_candidates = [torch.bfloat16, torch.float16]
        attn_candidates = ["flash_attention_2", "sdpa"]

        # FlashAttention 2 -> SDPA -> Default attention 순서로 시도
        for impl in attn_candidates:
            for dtype in dtype_candidates:
                try:
                    print(f"🧠 {impl} 어텐션과 {dtype} 데이터 타입으로 Qwen-VL 로드 시도 중...")
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        vl_model_name,
                        torch_dtype=dtype,
                        attn_implementation=impl,
                        device_map=device_map or "cuda",
                        low_cpu_mem_usage=True, # GPU 메모리 부족 시 CPU로 오프로드 시도
                    )
                    print(f"✅ Qwen-VL 모델 로드 성공: {impl} 어텐션 ({dtype})")
                    self.attn_backend = impl
                    self.model_dtype = dtype
                    return model
                except Exception as e:
                    print(f"⚠️ {impl} 어텐션 ({dtype}) 로드 실패: {e}")

        # 모든 특정 어텐션 구현 시도 실패 시, 기본 어텐션으로 재시도
        for dtype in dtype_candidates:
            try:
                print(f"🧠 기본 어텐션과 {dtype} 데이터 타입으로 Qwen-VL 로드 시도 중...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    vl_model_name,
                    torch_dtype=dtype,
                    device_map=device_map or "cuda",
                    low_cpu_mem_usage=True,
                )
                print(f"✅ Qwen-VL 모델 로드 성공: 기본 어텐션 ({dtype})")
                self.attn_backend = "default"
                self.model_dtype = dtype
                return model
            except Exception as e:
                print(f"⚠️ 기본 어텐션 ({dtype}) 로드 실패: {e}")

        raise RuntimeError("❌ 모든 Qwen-VL 모델 로드 시도 실패. 호환되는 설정이 없습니다.")

    def set_cache_enabled(self, enabled: bool = True):
        """내부 VL 특징 캐싱 활성화 여부를 `vl_encoder`에 위임합니다."""
        if hasattr(self, 'vl_encoder'):
            self.vl_encoder.set_cache_enabled(enabled)

    def set_strict_cache(self, enabled: bool = True):
        """내부 VL 특징 캐싱 시 엄격 모드를 `vl_encoder`에 위임합니다."""
        if hasattr(self, 'vl_encoder'):
            self.vl_encoder.set_strict_cache(enabled)

    def set_cache_limit_gb(self, limit_gb: float):
        """내부 VL 특징 캐시의 최대 크기를 `vl_encoder`에 위임합니다."""
        if hasattr(self, 'vl_encoder'):
            self.vl_encoder.set_cache_limit_gb(limit_gb)

    @staticmethod
    def _infer_cached_vl_hidden_size(cache_root: Optional[str]) -> Optional[int]:
        """외부 캐시에서 VL hidden size 추론 (cache_only_mode 전용)."""
        if not cache_root:
            return None

        cache_root_path = Path(cache_root)
        if not cache_root_path.exists():
            return None

        try:
            prompt_dirs = [d for d in cache_root_path.iterdir() if d.is_dir()]
        except Exception:
            return None

        for prompt_dir in prompt_dirs:
            try:
                cache_files = sorted(prompt_dir.glob("*.pt"))
            except Exception:
                continue

            for cache_file in cache_files:
                try:
                    cached = torch.load(cache_file, map_location="cpu")
                except Exception:
                    continue

                if isinstance(cached, tuple) and len(cached) == 2:
                    img_tokens, txt_tokens = cached
                    if isinstance(txt_tokens, torch.Tensor) and txt_tokens.shape[-1] > 0:
                        return int(txt_tokens.shape[-1])
                    if isinstance(img_tokens, torch.Tensor) and img_tokens.dim() >= 3 and img_tokens.shape[-1] > 0:
                        return int(img_tokens.shape[-1])
                elif isinstance(cached, torch.Tensor) and cached.dim() >= 2:
                    if cached.shape[-1] > 0:
                        return int(cached.shape[-1])

        return None

    def _prepare_dataloader_cached_vl_tokens(self, cached_batch: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]], device: torch.device) -> Tuple[Optional[Tuple[Union[torch.Tensor, List], Union[torch.Tensor, List]]], Optional[List[int]]]:
        """
        데이터로더에서 제공된 캐시된 VL 튜플(이미지, 텍스트)을 준비합니다.
        부분적인 캐시 커버리지를 처리합니다.
        """
        if not cached_batch:
            return None, None

        target_dtype = getattr(self, "model_dtype", torch.bfloat16)
        prepared_img_tokens: List[Optional[torch.Tensor]] = []
        prepared_txt_tokens: List[Optional[torch.Tensor]] = []
        missing_indices: List[int] = []
        has_any_valid_tensor = False

        for idx, item in enumerate(cached_batch):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                img_t, txt_t = item
                img_is_tensor = isinstance(img_t, torch.Tensor) and img_t.dim() == 3
                txt_is_tensor = isinstance(txt_t, torch.Tensor) and txt_t.numel() > 0

                if img_is_tensor and txt_is_tensor:
                    prepared_img_tokens.append(img_t.to(device=device, dtype=target_dtype, non_blocking=True))
                    prepared_txt_tokens.append(txt_t.to(device=device, dtype=target_dtype, non_blocking=True))
                    has_any_valid_tensor = True
                    continue

            prepared_img_tokens.append(None)
            prepared_txt_tokens.append(None)
            missing_indices.append(idx)

        if not has_any_valid_tensor:
            return None, None

        if not missing_indices:
            # 모든 샘플이 캐시됨
            return (torch.cat(prepared_img_tokens, dim=0), torch.cat(prepared_txt_tokens, dim=0)), None

        # 일부 샘플만 캐시됨
        return (prepared_img_tokens, prepared_txt_tokens), missing_indices

    def _encode_missing_vl_features_and_backfill(self,
                                                text_inputs: List[str],
                                                image_inputs: List[List[str]],
                                                cache_keys: List[str],
                                                indices_to_encode: List[int],
                                                device: torch.device,
                                                vl_cache_metadata: Optional[dict] = None) -> dict:
        """
        누락된 VL 특징(이미지, 텍스트 튜플)만을 인코딩하고 결과를 다시 채웁니다.
        """
        if not indices_to_encode:
            return {}

        subset_texts = [text_inputs[i] for i in indices_to_encode]
        subset_images = [image_inputs[i] for i in indices_to_encode]
        subset_keys = [cache_keys[i] for i in indices_to_encode]

        # VL 특징 인코딩 (V2: 튜플 반환)
        image_features, guidance_vectors = self.vl_encoder.encode(
            subset_texts, subset_images, subset_keys, use_cache=False
        )

        # 결과를 인덱스별 튜플로 분할
        img_splits = torch.split(image_features, 1, dim=0)
        txt_splits = torch.split(guidance_vectors, 1, dim=0)
        tokens_by_index = {
            idx: (img, txt) for idx, img, txt in zip(indices_to_encode, img_splits, txt_splits)
        }

        if self.auto_cache_backfill and self.external_cache_mgr and vl_cache_metadata:
            dataset_names = vl_cache_metadata.get("dataset_names")
            vlm_indices = vl_cache_metadata.get("vlm_indices")
            prompt_hashes = vl_cache_metadata.get("prompt_hashes")

            if dataset_names and vlm_indices and prompt_hashes:
                for idx, (img_tensor, txt_tensor) in tokens_by_index.items():
                    if img_tensor is None or txt_tensor is None: continue
                    try:
                        if idx >= len(dataset_names) or idx >= len(vlm_indices) or idx >= len(prompt_hashes): continue
                        dataset_name, vlm_idx, prompt_hash = dataset_names[idx], int(vlm_indices[idx]), prompt_hashes[idx]
                    except (TypeError, ValueError, IndexError):
                        continue
                    if dataset_name is None or prompt_hash is None: continue
                    
                    try:
                        if not hasattr(self, "_cache_backfill_notice"):
                            print("🧷 훈련 중 누락된 VL 캐시 항목을 자동으로 빌드합니다 (V2 형식).")
                            self._cache_backfill_notice = True
                        # V2: 튜플을 저장
                        self.external_cache_mgr.save_cache(
                            dataset_name=dataset_name, vlm_idx=vlm_idx, prompt_hash=prompt_hash,
                            vl_features=(img_tensor.detach(), txt_tensor.detach()),
                        )
                    except Exception as e:
                        if not hasattr(self, "_cache_backfill_warned"):
                            print(f"⚠️ VL 캐시 항목 백필 실패 ({dataset_name}_vlm{vlm_idx}): {e}")
                            self.backfill_warned = True
        return tokens_by_index

    def encode_vision(self,
                      text_inputs: List[str],
                      image_inputs: List[List[str]],
                      cache_keys: List[str],
                      use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        주어진 텍스트 및 이미지 입력으로부터 VL 특징(이미지, 텍스트 튜플)을 인코딩합니다.
        주로 캐시 빌딩 목적으로 사용됩니다.
        """
        self.eval()
        return self.vl_encoder.encode(
            text_inputs, image_inputs, cache_keys, use_cache=use_cache
        )

    def forward(self,
                text_inputs: List[str],
                image_inputs: List[List[str]],
                actions: Optional[torch.Tensor] = None,
                z_chunk: Optional[torch.Tensor] = None,
                sensor_data: Optional[torch.Tensor] = None,
                robot_states: Optional[torch.Tensor] = None,
                cache_keys: Optional[List[str]] = None,
                cache: bool = True,
                vl_cache_tokens: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
                vl_cache_metadata: Optional[dict] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        V2 아키텍처를 위한 통합 포워드 패스.
        이미지 특징과 텍스트 가이던스를 분리하여 행동 전문가에게 전달합니다.
        """
        device = next(self.parameters()).device
        image_features, guidance_vectors = None, None

        # 1. VL 특징 인코딩 및 캐싱 처리 (V2)
        prepared_cached_tokens, missing_indices = self._prepare_dataloader_cached_vl_tokens(vl_cache_tokens, device)

        if prepared_cached_tokens and not missing_indices:
            image_features, guidance_vectors = prepared_cached_tokens
            if not hasattr(self, "_external_cache_confirmed"):
                print("💾 데이터로더에서 제공된 VL 캐시 텐서(V2) 사용 중.")
                self._external_cache_confirmed = True
        elif prepared_cached_tokens and missing_indices:
            if self.cache_only_mode:
                raise RuntimeError("⚠️ cache_only_mode에서는 모든 VL 캐시가 준비되어 있어야 합니다. 누락된 캐시가 있습니다.")

            prepared_img_tokens, prepared_txt_tokens = prepared_cached_tokens
            new_tokens_by_idx = self._encode_missing_vl_features_and_backfill(
                text_inputs, image_inputs, cache_keys, missing_indices, device, vl_cache_metadata
            )
            for idx in missing_indices:
                if idx in new_tokens_by_idx:
                    prepared_img_tokens[idx], prepared_txt_tokens[idx] = new_tokens_by_idx[idx]

            if all(t is not None for t in prepared_img_tokens) and all(t is not None for t in prepared_txt_tokens):
                image_features = torch.cat(prepared_img_tokens, dim=0)
                guidance_vectors = torch.cat(prepared_txt_tokens, dim=0)
            else:
                raise RuntimeError("⚠️ 데이터로더 캐시와 신규 인코딩 후에도 VL 토큰이 완전히 준비되지 않았습니다.")
        else:
            if self.cache_only_mode:
                if actions is not None and self.training:
                    return torch.tensor(0.0, device=device, requires_grad=True), None, None
                else:
                    batch_size = len(text_inputs)
                    return torch.zeros(batch_size, self.horizon, self.action_dim, device=device), None, None

            image_features, guidance_vectors = self.vl_encoder.encode(
                text_inputs, image_inputs, cache_keys, use_cache=cache
            )

        # 2. 센서 및 로봇 상태 특징 인코딩
        sensor_features_encoded: Optional[torch.Tensor] = None
        if self.sensor_enabled and sensor_data is not None:
            sensor_features_encoded = self.sensor_encoder(sensor_data.to(device=device, dtype=torch.bfloat16))

        robot_state_features_encoded: Optional[torch.Tensor] = None
        if self.robot_state_enabled and robot_states is not None:
            robot_state_features_encoded = self.robot_state_encoder(robot_states.to(device=device, dtype=torch.bfloat16))

        # 3. 센서 특징 결합
        sensor_tensors = []
        if sensor_features_encoded is not None:
            sensor_tensors.append(sensor_features_encoded)
        if robot_state_features_encoded is not None:
            sensor_tensors.append(robot_state_features_encoded)

        sensor_features_combined: Optional[torch.Tensor] = None
        if sensor_tensors:
            if len(sensor_tensors) > 1:
                sensor_features_combined = torch.cat(sensor_tensors, dim=-1)
            else:
                sensor_features_combined = sensor_tensors[0]

        if image_features is not None and image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)

        # 4. 모델 타입에 따른 포워드 패스
        if self.model_type == 'flow_matching':
            if actions is not None and self.training:
                actions = actions.to(device=device, dtype=image_features.dtype)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    loss = self.action_expert.compute_loss(
                        actions, image_features, guidance_vectors,
                        sensor_features=sensor_features_combined
                    )
                return loss, None, None
            else:
                sampled_actions = self.action_expert.sample(
                    image_features, guidance_vectors,
                    sensor_features=sensor_features_combined,
                    num_steps=self.flow_steps, method=self.flow_solver
                )
                return sampled_actions, None, None
        elif self.model_type == 'regression':
            if z_chunk is None:
                raise ValueError("Regression 모델은 z_chunk 입력이 필요합니다.")
            z_chunk = z_chunk.to(device=device, dtype=image_features.dtype)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                pred_actions, delta = self.action_expert(
                    z_chunk, image_features, guidance_vectors,
                    sensor_features=sensor_features_combined
                )
            return pred_actions, delta
        else:
            raise ValueError(f"알 수 없는 모델 타입: {self.model_type}")

    @torch.no_grad()
    def predict_action(self,
                       text_inputs: List[str],
                       image_inputs: List[List[str]],
                       sensor_data: Optional[torch.Tensor] = None,
                       robot_states: Optional[torch.Tensor] = None,
                       cache_keys: Optional[List[str]] = None,
                       **kwargs) -> torch.Tensor:
        """
        V2 아키텍처를 위한 추론 전용 래퍼 함수.
        """
        self.eval()
        
        if self.model_type == 'flow_matching':
            sampled_actions, _, _ = self.forward(
                text_inputs=text_inputs, image_inputs=image_inputs,
                actions=None, sensor_data=sensor_data, robot_states=robot_states,
                cache_keys=cache_keys, **kwargs
            )
            return sampled_actions
        else:
            raise NotImplementedError(
                "Regression 모델의 predict_action은 z_chunk가 필요합니다. "
                "`forward()` 메서드를 `z_chunk` 매개변수와 함께 직접 사용해주세요."
            )

if __name__ == "__main__":
    print("🧪 Unified VLA 모델 V3 테스트 시작...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 가능한 장치: {device}")

    try:
        model_flow_matching = QwenVLAUnified(
            model_type='flow_matching', sensor_enabled=True, robot_state_enabled=True,
            finetune_vl='none', cache_dir="./test_cache",
        ).to(device)
        model_flow_matching.eval()

        batch_size = 2
        horizon = model_flow_matching.horizon
        action_dim = model_flow_matching.action_dim
        
        text_inputs_dummy = ["로봇 팔을 움직여 컵을 잡으시오.", "빨간 블록을 왼쪽으로 옮기시오."]
        import glob
        sample_images = glob.glob("/home/najo/NAS/VLA/dataset/New_dataset2/**/View4/*.jpg", recursive=True)[:4]
        if len(sample_images) >= 2:
            image_inputs_dummy = [[sample_images[0], sample_images[1]], [sample_images[2], sample_images[3]]]
            print(f"  테스트용 실제 이미지 사용: {len(image_inputs_dummy)}개 샘플, 각 2개 뷰")
        else:
            print("  ⚠️ 실제 이미지를 찾을 수 없어 빈 리스트로 테스트합니다.")
            image_inputs_dummy = [[], []]

        sensor_data_dummy = torch.randn(batch_size, 65, 1026, device=device, dtype=torch.float32)
        robot_states_dummy = torch.randn(batch_size, 100, 12, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            sampled_actions_flow = model_flow_matching.predict_action(
                text_inputs=text_inputs_dummy, image_inputs=image_inputs_dummy,
                sensor_data=sensor_data_dummy, robot_states=robot_states_dummy
            )
        print(f"✅ Flow Matching 모델 추론 성공. 출력 형태: {sampled_actions_flow.shape}")
        assert sampled_actions_flow.shape == (batch_size, horizon, action_dim)
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ 모든 테스트 완료!")
    
    test_cache_dir = Path("./test_cache")
    if test_cache_dir.exists():
        import shutil
        print(f"테스트 캐시 디렉토리 {test_cache_dir} 정리 중...")
        shutil.rmtree(test_cache_dir)
        print("정리 완료.")
