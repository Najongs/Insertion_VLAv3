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
    - `SensorEncoder`: A temporal ConvFormer for general time-series sensor data.
    - `ForceAwareSensorEncoder`: Specializes in processing distance and force features separately.
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
#    - 센서 인코더 (SensorEncoder) 및 Force-Aware 센서 인코더
# ==============================================================================

class RobotStateEncoder(nn.Module):
    """
    로봇의 관절 각도(joint angles)와 엔드 이펙터 포즈(end-effector pose) 시퀀스를 처리하는
    Transformer 기반 인코더입니다. Temporal Pooling과 Projection을 통해 고정된 크기의
    출력 특징 벡터를 생성합니다.
    """
    def __init__(self,
                 input_dim: int = 12, # 6 관절 각도 + 6 엔드 이펙터 포즈 (x, y, z, roll, pitch, yaw)
                 model_dim: int = 256,
                 output_dim: int = 2048,
                 num_heads: int = 8,
                 num_layers: int = 4,
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

class SensorEncoder(nn.Module):
    """
    향상된 센서 인코더 (Temporal ConvFormer).
    1D Convolutional 백본과 Transformer 인코더를 결합하여 시계열 센서 데이터를 처리합니다.

    Args:
        input_channels (int, optional): 입력 센서 데이터의 채널 수. Defaults to 1026.
        temporal_length (int, optional): 입력 시계열 데이터의 예상 길이. Defaults to 650.
        hidden_dim (int, optional): 컨볼루션 레이어의 초기 은닉 차원. Defaults to 512.
        output_dim (int, optional): 최종 출력 특징 벡터의 차원. Defaults to 3072.
        num_conv_layers (int, optional): 컨볼루션 백본의 레이어 수. Defaults to 4.
        use_transformer (bool, optional): 컨볼루션 후 트랜스포머를 사용할지 여부. Defaults to True.
        num_transformer_layers (int, optional): 트랜스포머 인코더 레이어 수. Defaults to 2.
        nhead (int, optional): 트랜스포머의 어텐션 헤드 수. Defaults to 8.
        dropout (float, optional): 드롭아웃 비율. Defaults to 0.1.
        gradient_checkpointing (bool, optional): 메모리 절약을 위한 그래디언트 체크포인팅 활성화 여부. Defaults to False.
        interpolation_mode (str, optional): 시계열 길이 불일치 시 보간 모드 ('linear', 'cubic', 'nearest'). Defaults to 'linear'.
    """
    def __init__(self,
                 input_channels=1026,
                 temporal_length=650,
                 hidden_dim=512,
                 output_dim=3072,
                 num_conv_layers=4,
                 use_transformer=True,
                 num_transformer_layers=2,
                 nhead=8,
                 dropout=0.1,
                 gradient_checkpointing=False,
                 interpolation_mode='linear'):
        super().__init__()
        self.input_channels = input_channels
        self.temporal_length = temporal_length
        self.output_dim = output_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.interpolation_mode = interpolation_mode

        # 잔차 다운샘플 블록 스택 (Residual Downsample Block Stack)
        # 센서 데이터의 채널을 확장하고 시간 차원을 점진적으로 줄입니다.
        chs = [input_channels]
        conv_blocks = []
        for i in range(num_conv_layers):
            out_ch = hidden_dim if i == 0 else hidden_dim * (2 ** i) # 채널 증가는 예전 코드 방식 유지
            conv_blocks.append(ResidualDownsample1d(
                in_ch=chs[-1], out_ch=out_ch, stride=2, dropout=dropout
            ))
            chs.append(out_ch)
        self.conv_backbone = nn.ModuleList(conv_blocks)
        self.final_channels = chs[-1] # 최종 컨볼루션 출력 채널

        # 최종 시간 길이 (대략 절반씩 줄어듦, 정확한 계산은 아님)
        # 실제 길이는 F.interpolate 후 컨볼루션 연산으로 인해 달라질 수 있음.
        self.final_temporal_length = temporal_length // (2 ** num_conv_layers) # 근사값

        self.use_transformer = use_transformer
        if use_transformer:
            # 컨볼루션 백본의 출력 특징에 대해 Transformer를 적용하여 장거리 시간적 의존성 학습
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.final_channels, nhead=nhead,
                dim_feedforward=self.final_channels * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        # Temporal Pooling 및 Projection: 처리된 시계열 특징을 고정 길이 벡터로 변환
        self.temporal_pool = nn.AdaptiveAvgPool1d(1) # 시간 축 평균 풀링
        self.projection = nn.Sequential(
            nn.Linear(self.final_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        센서 데이터를 인코딩하여 특징 벡터를 반환합니다.

        Args:
            sensor_data (torch.Tensor): 입력 센서 데이터, (B, T, C) 형태.

        Returns:
            torch.Tensor: 인코딩된 센서 특징, (B, output_dim) 형태.
        """
        B, T, C = sensor_data.shape
        if C != self.input_channels:
            raise ValueError(f"예상되는 채널 수 {self.input_channels}와 다릅니다. 현재: {C}")

        # 비동기 길이 보정 (interpolation)
        # 입력 시계열 길이가 모델이 예상하는 길이와 다르면 보간을 수행합니다.
        if T != self.temporal_length:
            x = sensor_data.transpose(1, 2)  # (B, C, T) 형태로 변경
            # 인터폴레이션 모드 선택: 너무 짧은 시퀀스에는 'cubic' 대신 'linear' 사용
            mode = 'linear' if self.interpolation_mode == 'cubic' and T < 4 else self.interpolation_mode
            x = F.interpolate(x, size=self.temporal_length, mode=mode,
                              align_corners=False if mode in ('linear', 'cubic') else None)
        else:
            x = sensor_data.transpose(1, 2)  # (B, C, T) 형태로 변경

        # Conv 백본 통과
        for block in self.conv_backbone:
            if self.gradient_checkpointing and self.training:
                # 메모리 절약을 위한 그래디언트 체크포인팅
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)   # (B, ch, T')

        # Transformer 통과 (사용 활성화 시)
        if self.use_transformer:
            x = x.transpose(1, 2)  # (B, T', ch) 형태로 변경하여 트랜스포머에 입력
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
            else:
                x = self.transformer(x)
            x = x.transpose(1, 2)  # 다시 (B, ch, T') 형태로 변경

        # 시계열 풀링 및 최종 투영
        x = self.temporal_pool(x).squeeze(-1)        # (B, ch)
        sensor_features = self.projection(x)         # (B, output_dim)
        return sensor_features

class ForceAwareSensorEncoder(nn.Module):
    """
    '거리' (distance)와 '힘' (force) 특징을 개별적으로 처리하여
    힘 데이터에 더 많은 가중치를 부여하는 센서 인코더입니다.

    아키텍처:
    1.  주요 '거리' 특징(`dist_channels`)은 표준 `SensorEncoder`로 처리됩니다.
    2.  '힘' 특징(`force_channels`)은 전용 MLP로 처리됩니다.
    3.  두 인코더의 출력이 결합(concat)되어 최종 출력 차원으로 투영됩니다.

    Args:
        dist_channels (int, optional): 거리 센서 채널 수. Defaults to 1025.
        force_channels (int, optional): 힘 센서 채널 수. Defaults to 1.
        temporal_length (int, optional): 시계열 길이. Defaults to 65.
        dist_hidden_dim (int, optional): 거리 인코더의 은닉 차원. Defaults to 512.
        force_hidden_dim (int, optional): 힘 인코더의 은닉 차원. Defaults to 128.
        output_dim (int, optional): 최종 출력 특징 벡터의 차원. Defaults to 3072.
        **kwargs: `SensorEncoder`로 전달될 추가 인자들.
    """
    def __init__(self,
                 dist_channels=1025,
                 force_channels=1,
                 temporal_length=65,
                 dist_hidden_dim=512,
                 force_hidden_dim=128,
                 output_dim=3072,
                 **kwargs):
        super().__init__()
        self.input_channels = dist_channels + force_channels
        self.force_channels = force_channels # forward에서 사용하기 위해 저장

        print(f"🚀 ForceAwareSensorEncoder 초기화 중:")
        print(f"   - 거리 특징 (1-{dist_channels})은 ConvFormer로 처리.")
        print(f"   - 힘 특징 ({dist_channels+1})은 전용 MLP로 처리.")

        # 주요 '거리' 특징을 위한 인코더
        self.dist_encoder = SensorEncoder(
            input_channels=dist_channels,
            temporal_length=temporal_length,
            hidden_dim=dist_hidden_dim,
            output_dim=output_dim - force_hidden_dim, # 출력 공간의 일부를 할당
            **kwargs
        )
        # `dist_encoder` 내의 BatchNorm 레이어를 float32로 강제
        force_bn_fp32_(self.dist_encoder)

        # '힘' 특징을 위한 작고 전용 MLP
        self.force_encoder = nn.Sequential(
            nn.Linear(force_channels, force_hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(force_hidden_dim // 2),
            nn.Linear(force_hidden_dim // 2, force_hidden_dim)
        )
        self.force_pool = nn.AdaptiveAvgPool1d(1) # 시간 축 풀링

        print(f"   - 최종 출력: {output_dim} 차원으로 결합 및 투영.")


    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Force-Aware 방식으로 센서 데이터를 인코딩합니다.

        Args:
            sensor_data (torch.Tensor): 입력 센서 데이터, (B, T, C) 형태 (여기서 C는 self.input_channels).

        Returns:
            torch.Tensor: 인코딩된 결합 특징, (B, output_dim) 형태.
        """
        B, T, C = sensor_data.shape
        if C != self.input_channels:
            raise ValueError(f"예상되는 채널 수 {self.input_channels}와 다릅니다. 현재: {C}")

        # 데이터를 거리와 힘으로 분리
        dist_data = sensor_data[..., :-self.force_channels]  # (B, T, dist_channels)
        force_data = sensor_data[..., -self.force_channels:] # (B, T, force_channels)

        # 1. 거리 데이터 처리
        dist_features = self.dist_encoder(dist_data) # (B, output_dim - force_hidden_dim)

        # 2. 힘 데이터 처리 (MLP -> 시간 풀링)
        force_features_temporal = self.force_encoder(force_data) # (B, T, force_hidden_dim)
        force_features_pooled = self.force_pool(force_features_temporal.transpose(1, 2)).squeeze(-1) # (B, force_hidden_dim)

        # 3. 두 특징 결합 및 반환
        combined_features = torch.cat([dist_features, force_features_pooled], dim=-1)

        return combined_features
