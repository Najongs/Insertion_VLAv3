"""
VLA Encoder Modules for Robot State and Sensor Data.

This file defines various encoder modules used within the unified VLA model
to process robot proprioceptive states and diverse sensor data.

Components:
- Helper Functions/Classes:
    - `ResidualDownsample1d`: 1D Residual Downsampling block for time-series data.
    - `force_bn_fp32_`: Utility to cast BatchNorm layers to float32.
- Encoder Modules:
    - `RobotStateEncoder`: Processes sequences of robot joint angles and end-effector poses.
    - `UnifiedGatedSensorEncoder`: An advanced sensor encoder that fuses distance and
      force data using a gated, asymmetric attention mechanism.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. 헬퍼 함수 및 클래스 (Helper Functions & Classes)
# ==============================================================================

class ResidualDownsample1d(nn.Module):
    """
    1D Residual Downsampling 블록. 시계열 데이터 처리에 사용됩니다.
    BatchNorm 레이어는 항상 FP32로 고정됩니다.
    """
    def __init__(self, in_ch, out_ch, stride=2, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.act1  = nn.GELU()
        self.do1   = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act2  = nn.GELU()

        self.skip  = nn.Identity() if (in_ch == out_ch and stride == 1) else \
                     nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

        # BatchNorm을 항상 FP32로 고정
        self.bn1.float()
        self.bn2.float()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.do1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)

        s = self.skip(x)
        return y + s


def force_bn_fp32_(module: torch.nn.Module):
    """
    주어진 모듈 내의 모든 BatchNorm 레이어의 매개변수/버퍼를 float32로 캐스팅합니다.
    혼합 정밀도(Mixed Precision) 훈련 시 주의 사항.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()  # 가중치/편향 및 러닝 통계 모두 FP32로

# ==============================================================================
# 2. 인코더 모듈 (Encoder Modules)
#    - 로봇 상태 인코더 (RobotStateEncoder)
#    - 통합 게이트 센서 인코더 (UnifiedGatedSensorEncoder)
# ==============================================================================

class RobotStateEncoder(nn.Module):
    """
    로봇의 관절 각도(joint angles)와 엔드 이펙터 포즈(end-effector pose) 시퀀스를 처리하는
    Transformer 기반 인코더입니다. Temporal Pooling과 Projection을 통해 고정된 크기의
    출력 특징 벡터를 생성합니다.
    """
    def __init__(self,
                input_dim: int = 12, # 6 관절 각도 + 6 엔드 이펙터 포즈 (x, y, z, roll, pitch, yaw)
                model_dim: int = 512,
                output_dim: int = 1024,
                num_heads: int = 8,
                num_layers: int = 6,
                temporal_length: int = 60,
                dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim

        # 1. 입력 투영 (Input Projection): 원시 로봇 상태를 모델 차원으로 매핑
        self.input_proj = nn.Linear(input_dim, model_dim)

        # 2. 위치 인코딩 (Positional Encoding): 시계열 데이터의 시간적 순서 정보 제공
        self.pos_encoder = nn.Parameter(torch.zeros(1, temporal_length, model_dim))

        # 3. Transformer 인코더: 시간적 의존성을 학습
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True, # (Batch, Sequence, Feature) 형태로 입력 처리
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Temporal Pooling 및 Projection Head: 시계열 특징을 고정 길이 벡터로 압축하고 최종 출력 차원으로 매핑
        self.temporal_pool = nn.AdaptiveAvgPool1d(1) # 시간 축 평균 풀링
        self.projection = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        로봇 상태 시퀀스를 인코딩합니다.

        Args:
            src (torch.Tensor): 로봇 상태 시퀀스, (B, T, D_in) 형태.
            return_sequence (bool): True이면 트랜스포머의 전체 출력 시퀀스를 반환하고,
                                    False (기본값)이면 풀링 및 투영된 특징 벡터를 반환합니다.

        Returns:
            torch.Tensor: 인코딩된 특징.
                          return_sequence가 False이면 (B, D_out) 형태,
                          True이면 (B, T, model_dim) 형태입니다.
        """
        # 입력 투영 및 위치 인코딩 추가
        x = self.input_proj(src) # (B, T, model_dim)
        x = x + self.pos_encoder # 위치 인코딩 더하기
        x = self.dropout(x)

        # 트랜스포머 통과
        x = self.transformer_encoder(x) # (B, T, model_dim)

        # MAE 사전 훈련과 같이 시퀀스 자체를 반환해야 하는 경우
        if return_sequence:
            return x

        # 다운스트림 작업을 위한 풀링 및 투영
        pooled_x = x.transpose(1, 2) # (B, model_dim, T) 형태로 변경하여 1D 풀링 준비
        pooled_x = self.temporal_pool(pooled_x).squeeze(-1) # (B, model_dim)
        output_features = self.projection(pooled_x) # (B, output_dim)

        return output_features


class UnifiedGatedSensorEncoder(nn.Module):
    """
    Distance와 Force 센서 데이터를 지능적으로 융합하는 통합 인코더.
    bfloat16 기준 약 50MB 크기로 설계되었습니다.

    주요 특징:
    1.  **비대칭 처리**: Distance는 Conv-Transformer, Force는 MLP로 처리하여 중요도 차등 반영.
    2.  **통합 유효성 게이트**: 두 센서 정보를 종합하여 현재 데이터의 유효성을 판단,
        무의미한 신호(예: 물체에서 너무 멀리 떨어져 있을 때)를 동적으로 필터링.
    3.  **게이트 적용 융합**: Cross-Attention을 통한 정보 융합이 유효성 게이트 값에 따라
        조절되어, 필요할 때만 Force 정보가 Distance 정보를 보강.
    """
    def __init__(self,
                 dist_channels=1025,
                 force_channels=1,
                 temporal_length=65,
                 # bfloat16 기준 ~50MB (약 26.5M 파라미터) 목표 하이퍼파라미터
                 conv_hidden_dim=256,
                 num_conv_layers=4,
                 model_dim=512,
                 num_transformer_layers=4,
                 nhead=8,
                 force_mlp_dim=256,
                 output_dim=3072,
                 dropout=0.1,
                 gradient_checkpointing=False):
        super().__init__()
        self.input_channels = dist_channels + force_channels
        self.force_channels = force_channels
        self.temporal_length = temporal_length
        self.gradient_checkpointing = gradient_checkpointing

        # --- 1. Distance Branch (주요 파이프라인) ---
        dist_conv_chs = [dist_channels]
        dist_conv_blocks = []
        current_ch = dist_channels
        for i in range(num_conv_layers):
            out_ch = conv_hidden_dim * (2 ** i)
            dist_conv_blocks.append(ResidualDownsample1d(current_ch, out_ch, stride=2, dropout=dropout))
            current_ch = out_ch
        self.dist_conv_backbone = nn.ModuleList(dist_conv_blocks)

        self.conv_to_transformer_proj = nn.Conv1d(current_ch, model_dim, kernel_size=1)

        dist_transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.dist_transformer = nn.TransformerEncoder(dist_transformer_layer, num_layers=num_transformer_layers)
        self.dist_norm = nn.LayerNorm(model_dim)

        # --- 2. Force Branch (보조 파이프라인) ---
        self.force_encoder = nn.Sequential(
            nn.Linear(force_channels, force_mlp_dim),
            nn.GELU(),
            nn.LayerNorm(force_mlp_dim),
            nn.Linear(force_mlp_dim, model_dim) # Cross-Attention을 위해 model_dim으로 출력
        )

        # --- 3. 통합 유효성 게이트 (Unified Validity Gate) ---
        gate_input_dim = model_dim + model_dim # dist_summary + force_summary
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_input_dim // 4),
            nn.GELU(),
            nn.Linear(gate_input_dim // 4, 1)
        )
        self.gate_activation = nn.Sigmoid()

        # --- 4. 게이트 적용 융합 (Gated Fusion Block) ---
        self.fusion_cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(model_dim)

        # --- 5. 최종 출력 (Final Projection) ---
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        force_bn_fp32_(self) # 모든 BatchNorm 레이어를 FP32로 유지

    def _run_block(self, block, x):
        """그래디언트 체크포인팅을 위한 래퍼 함수"""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, sensor_data: torch.Tensor, return_sequence: bool = False):
        """
        Args:
            sensor_data (torch.Tensor): (B, T, C) 형태의 센서 데이터
            return_sequence (bool): 시퀀스 특징과 전역 특징을 모두 반환할지 여부

        Returns:
            - False (기본값): (B, output_dim) 형태의 전역 특징
            - True: ((B, T', model_dim), (B, output_dim)) 형태의 튜플
        """
        B, T, C = sensor_data.shape
        if C != self.input_channels:
            raise ValueError(f"Input channels {C} does not match expected {self.input_channels}")

        # --- 데이터 전처리 및 분리 ---
        # 길이 보간
        if T != self.temporal_length:
            x = sensor_data.transpose(1, 2) # (B, C, T)
            x = F.interpolate(x, size=self.temporal_length, mode='linear', align_corners=False)
        else:
            x = sensor_data.transpose(1, 2) # (B, C, T)

        dist_data = x[:, :-self.force_channels, :]  # (B, dist_C, T)
        force_data = x[:, -self.force_channels:, :] # (B, force_C, T)
        force_data = force_data.transpose(1, 2)     # (B, T, force_C)

        # --- 1. Distance Branch ---
        dist_seq = dist_data
        for block in self.dist_conv_backbone:
            dist_seq = self._run_block(block, dist_seq)
        dist_seq = self.conv_to_transformer_proj(dist_seq) # (B, model_dim, T')
        dist_seq = dist_seq.transpose(1, 2) # (B, T', model_dim)
        dist_seq = self.dist_norm(self._run_block(self.dist_transformer, dist_seq))

        # --- 2. Force Branch ---
        force_seq = self._run_block(self.force_encoder, force_data) # (B, T, model_dim)
        # Force 시퀀스 길이를 Distance 시퀀스 길이에 맞춤
        if force_seq.shape[1] != dist_seq.shape[1]:
            force_seq_t = force_seq.transpose(1, 2) # (B, model_dim, T)
            force_seq_t = F.interpolate(force_seq_t, size=dist_seq.shape[1], mode='linear', align_corners=False)
            force_seq = force_seq_t.transpose(1, 2) # (B, T', model_dim)

        # --- 3. 통합 유효성 게이트 ---
        dist_summary = torch.mean(dist_seq, dim=1)
        force_summary = torch.mean(force_seq, dim=1)
        gate_input = torch.cat([dist_summary, force_summary], dim=-1)
        gate_logit = self.gate_network(gate_input) # (B, 1)
        validity_gate = self.gate_activation(gate_logit).unsqueeze(-1) # (B, 1, 1)

        # --- 4. 게이트 적용 융합 ---
        # Cross-Attention: dist_seq가 force_seq로부터 정보를 가져옴
        force_context, _ = self.fusion_cross_attention(
            query=dist_seq, key=force_seq, value=force_seq
        )
        # 게이트 적용 및 잔차 연결
        gated_force_context = force_context * validity_gate
        fused_seq = self.fusion_norm(dist_seq + gated_force_context)

        # --- 5. 최종 출력 ---
        pooled_features = self.temporal_pool(fused_seq.transpose(1, 2)).squeeze(-1) # (B, model_dim)
        global_features = self.projection(pooled_features) # (B, output_dim)

        # Squeeze gate_logit to (B,)
        gate_logit = gate_logit.squeeze(-1)
        # gate_prob = self.gate_activation(gate_logit) # Reverted to output logits for BCEWithLogitsLoss

        if return_sequence:
            return fused_seq, global_features, gate_logit
        return global_features, gate_logit
