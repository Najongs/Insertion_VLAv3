"""
Action Decoder Models for VLA

이 파일은 행동 예측을 위한 다양한 디코더 모델(Action Experts)을 정의합니다.
- **FlowMatchingActionExpert**: Optimal Transport Conditional Flow Matching 기반
- **RegressionActionExpert**: 직접 회귀(Direct Regression) 기반
- **DiffusionActionExpert**: Denoising Diffusion Probabilistic Model 기반 (Deprecated)
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 헬퍼 함수 및 클래스
# ==============================================================================
class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        scale, shift = self.mod(cond).chunk(2, dim=-1)
        x = self.ln(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class ModulatedDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, dropout: float, ffn_mult: int = 4):
        super().__init__()
        self.self_ln = AdaLayerNorm(hidden_dim)
        self.cross_ln = AdaLayerNorm(hidden_dim)
        self.ffn_ln = AdaLayerNorm(hidden_dim)

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.last_cross_attn_weights: Optional[torch.Tensor] = None

    def forward(self, tgt, memory, cond, tgt_mask=None, memory_key_padding_mask=None):
        x = self.self_ln(tgt, cond)
        sa, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)
        tgt = tgt + self.dropout(sa)

        x = self.cross_ln(tgt, cond)
        ca, attn_w = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_cross_attn_weights = attn_w.detach()
        tgt = tgt + self.dropout(ca)

        x = self.ffn_ln(tgt, cond)
        tgt = tgt + self.dropout(self.ffn(x))
        return tgt
    
def cosine_beta_schedule(timesteps, s=0.008):
    """
    코사인 스케줄에 따른 베타 값 계산.
    Diffusion Schedule에서 사용됩니다.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

class OptimalTransportConditionalFlowMatching(nn.Module):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) 구현.
    """
    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min

    def compute_flow_and_target(self, x0: torch.Tensor, x1: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        t = torch.rand(B, 1, 1, device=device, dtype=dtype).expand_as(x0)
        t_scalar = t[:, 0, 0]

        if x1 is None:
            z = torch.randn_like(x0)
            mean_t = t * x0 + (1 - t) * z
            std_t = self.sigma_min * (1 - t)
            et = torch.randn_like(x0)
            x_t = mean_t + std_t * et
            u_t = x0 - z - self.sigma_min * et
        else:
            x_t = t * x1 + (1 - t) * x0
            u_t = x1 - x0

        return x_t, u_t, t_scalar

    def sample_ode(
        self,
        velocity_fn: callable,
        x_0: torch.Tensor,
        num_steps: int = 10,
        method: str = 'euler'
    ) -> torch.Tensor:
        device = x_0.device
        dtype = x_0.dtype
        x = x_0
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

        for i in range(num_steps):
            t = timesteps[i]
            h = timesteps[i+1] - timesteps[i]
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=dtype)
            
            if method == 'euler':
                v = velocity_fn(x, t_batch)
                x = x + h * v
            elif method == 'rk4':
                k1 = velocity_fn(x, t_batch)
                k2 = velocity_fn(x + 0.5 * h * k1, t_batch + 0.5 * h)
                k3 = velocity_fn(x + 0.5 * h * k2, t_batch + 0.5 * h)
                k4 = velocity_fn(x + h * k3, t_batch + h)
                x = x + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"알 수 없는 ODE 통합 방법: {method}")
        return x

# ==============================================================================
# 2. 액션 전문가 모듈 (Action Expert Modules)
# ==============================================================================

class FlowMatchingActionExpert(nn.Module):
    """
    Flow Matching 기반 액션 전문가 모델 V2.
    이미지 특징 시퀀스에 대한 Cross-Attention을 수행하여 공간 정보를 활용합니다.
    """
    def __init__(
        self,
        image_feature_dim: int = 2048,
        text_guidance_dim: int = 2048,
        sensor_dim: int = 2048,
        action_dim: int = 7,
        horizon: int = 8,
        hidden_dim: int = 1024,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        time_embed_dim: int = 256,
        dropout: float = 0.1,
        sigma_min: float = 1e-4
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.flow = OptimalTransportConditionalFlowMatching(sigma_min=sigma_min)
        
        # --- 입력 프로젝션 레이어 ---
        self.text_guidance_proj = nn.Linear(text_guidance_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        
        # 시간/가이던스 정규화(안정화)
        self.time_norm = nn.LayerNorm(hidden_dim)
        self.guidance_norm = nn.LayerNorm(hidden_dim)
        
        # --- 시간 임베딩 ---
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # --- 핵심 아키텍처: Transformer Decoder ---
        self.num_decoder_layers = num_decoder_layers
        self.mod_layers = nn.ModuleList([
            ModulatedDecoderLayer(hidden_dim, nhead, dropout, ffn_mult=4)
            for _ in range(num_decoder_layers)
        ])


        # --- 출력 헤드 ---
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # --- (PATCH A1) Sensor 토큰/타입 임베딩 설정 ---
        self.use_sensor_tokens = True  # 센서 토큰을 메모리에 함께 넣음 (False로 끄기 가능)
        self.context_proj = nn.Linear(image_feature_dim, hidden_dim)  # 컨텍스트(비전) 프로젝션
        self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)          # 센서 벡터/시퀀스 프로젝션
        self.token_type_embed = nn.Embedding(2, hidden_dim)           # 0=vision, 1=sensor

        # 포지셔널(메모리/타깃) 임베딩
        self.mem_pos = nn.Parameter(torch.randn(1, 512, hidden_dim))     # 메모리 최대길이 가정
        self.tgt_pos = nn.Parameter(torch.randn(1, horizon, hidden_dim)) # 타깃 길이=H
        
        # (PATCH B1) 디코더 자기어텐션 인과 마스크
        self.causal_self_attn = True
        
        # (PATCH D1) 액션 스케일/로스 가중
        self.action_scale = nn.Parameter(torch.tensor([10.,10.,10., 1.,1.,1., 1.]), requires_grad=False)  # 데이터 통계로 교체 권장
        self.min_lambda = 1e-3


        print(f"✅ FlowMatchingActionExpert V2 (Cross-Attention + ModulatedDecoder) 초기화 완료")
        print(f"   {num_decoder_layers}개의 ModulatedDecoderLayer 사용")


    def sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_embed_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb.to(t.dtype)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context_features: torch.Tensor,
        guidance_vector: torch.Tensor,
        sensor_features: Optional[torch.Tensor] = None,   # <-- 추가
    ) -> torch.Tensor:

        B, H, _ = x_t.shape
        
        assert context_features.dim() == 3, f"context_features (B,S,D) 필요, got {context_features.shape}"
        if self.use_sensor_tokens:
            assert sensor_features is not None, "sensor_features가 필요합니다(use_sensor_tokens=True)"

        # 1) 시간/가이던스 임베딩
        t_embed = self.time_mlp(self.sinusoidal_time_embedding(t))
        t_embed = self.time_norm(t_embed)  # 안정화
        guidance_embed = self.guidance_norm(self.text_guidance_proj(guidance_vector))
        cond_embed = guidance_embed + t_embed  # (B, D)

        # 2) 디코더 입력(tgt) + 포지셔널
        tgt = self.action_embed(x_t) + self.tgt_pos[:, :H]  # (B,H,D)

        # 3) 메모리(컨텍스트) 프로젝션 + 포지셔널
        # (PATCH A3) 메모리(비전+센서) 구성 + 포지셔널/타입 임베딩
        vision_mem = self.context_proj(context_features)  # (B, Sv, D)
        vision_mem = vision_mem + self.token_type_embed.weight[0].view(1, 1, -1)

        if self.use_sensor_tokens:
            assert sensor_features is not None, "use_sensor_tokens=True이면 sensor_features 필요"
            if sensor_features.dim() == 2:
                sensor_tok = self.sensor_proj(sensor_features).unsqueeze(1)  # (B,1,D)
            else:
                sensor_tok = self.sensor_proj(sensor_features)               # (B,Ss,D)
            sensor_tok = sensor_tok + self.token_type_embed.weight[1].view(1, 1, -1)

            Sv = vision_mem.size(1)
            Ss = sensor_tok.size(1)
            vision_mem = vision_mem + self.mem_pos[:, :Sv]
            sensor_tok = sensor_tok + (self.mem_pos[:, :Ss] if Ss <= self.mem_pos.size(1) else 0.0)

            memory = torch.cat([vision_mem, sensor_tok], dim=1)  # (B, Sv+Ss, D)
        else:
            Sv = vision_mem.size(1)
            memory = vision_mem + self.mem_pos[:, :Sv]


        # 4) 컨디셔닝 벡터를 tgt에 주입
        conditioned_tgt = tgt + cond_embed.unsqueeze(1)  # (B, H, D)
        
        tgt_mask = None
        if self.causal_self_attn:
            H_ = conditioned_tgt.size(1)
            tgt_mask = torch.triu(torch.ones(H_, H_, device=conditioned_tgt.device), diagonal=1).bool()

        # 5) Transformer Decoder
        x = conditioned_tgt
        for i in range(self.num_decoder_layers):
            x = self.mod_layers[i](x, memory, cond_embed, tgt_mask=tgt_mask)
        decoder_output = x


        # 6) 속도장 예측
        velocity = self.output_head(decoder_output)  # (B, H, action_dim)
        return velocity


    def compute_loss(
        self,
        actions: torch.Tensor,
        context_features: torch.Tensor,
        guidance_vector: torch.Tensor,
        sensor_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 스케일 정규화
        actions_n = actions / self.action_scale  # (B,H,A)

        x_t, u_t, t_scalar = self.flow.compute_flow_and_target(actions_n)
        v_pred = self.forward(x_t, t_scalar, context_features, guidance_vector, sensor_features=sensor_features)

        # λ(t) = (1 - t)
        lam = (1.0 - t_scalar).clamp_(min=0.0)
        lam = lam.view(lam.size(0), -1).mean(dim=1)  # (B,)
        lam = lam.view(-1, 1, 1).clamp(min=self.min_lambda)

        loss = F.smooth_l1_loss(v_pred, u_t, reduction='none')
        loss = (lam * loss).mean(dim=(1, 2)).mean()
        return loss


    @torch.no_grad()
    def sample(
        self,
        context_features: torch.Tensor,
        guidance_vector: torch.Tensor,
        sensor_features: Optional[torch.Tensor] = None,   # <-- 추가
        batch_size: Optional[int] = None,
        num_steps: int = 6,                               # 기본값 변경
        method: str = 'rk4'                               # 기본값 변경
    ) -> torch.Tensor:

        if batch_size is None:
            batch_size = context_features.shape[0]
        device = context_features.device
        
        x_0 = torch.randn(batch_size, self.horizon, self.action_dim, device=device, dtype=context_features.dtype)
        
        def velocity_fn(x, t):
            return self.forward(x, t, context_features, guidance_vector, sensor_features=sensor_features)
        
        actions = self.flow.sample_ode(velocity_fn, x_0, num_steps=num_steps, method=method)
        actions = actions * self.action_scale  # 원 단위 복원
        return actions

class RegressionActionExpert(nn.Module):
    """
    회귀 기반 액션 전문가 모델 V2.
    이미지 특징 시퀀스에 대한 Cross-Attention을 수행합니다.
    """
    def __init__(self,
                 image_feature_dim: int = 2048,
                 text_guidance_dim: int = 2048,
                 sensor_dim: int = 2048,
                 action_dim: int = 7,
                 horizon: int = 8,
                 hidden_dim: int = 1024,
                 nhead: int = 8,
                 num_decoder_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # --- 입력 프로젝션 레이어 ---
        self.text_guidance_proj = nn.Linear(text_guidance_dim, hidden_dim)
        
        # --- 컨텍스트(이미지 토큰) 프로젝션 & 포지셔널 임베딩 ---
        self.context_proj = nn.Linear(image_feature_dim, hidden_dim)
        self.mem_pos = nn.Parameter(torch.randn(1, 512, hidden_dim))  # 최대 길이 가정(슬라이스 사용)

        # --- 핵심 아키텍처: Transformer Decoder ---
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, hidden_dim))
        self.num_decoder_layers = num_decoder_layers
        self.mod_layers = nn.ModuleList([
            ModulatedDecoderLayer(hidden_dim, nhead, dropout, ffn_mult=4)
            for _ in range(num_decoder_layers)
        ])
        self.causal_self_attn = True  # 시계열 누설 방지 옵션

        # (PATCH F1) 델타 안전 한계
        self.max_delta = nn.Parameter(torch.tensor([5.,5.,5., 0.2,0.2,0.2, 1.]), requires_grad=False)
        self.token_type_embed = nn.Embedding(2, hidden_dim)  # 0=vision, 1=sensor
        self.use_sensor_tokens = True
        self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)

        
        # --- 출력 헤드 ---
        self.trans_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.rot_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.grip_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        
        print(f"✅ RegressionActionExpert V2 (Cross-Attention) 초기화 완료")

    def forward(
            self, 
            z_chunk: torch.Tensor,
            context_features: torch.Tensor,
            guidance_vector: torch.Tensor,
            sensor_features: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, _, _ = z_chunk.shape

        # 1. 컨디셔닝 벡터 생성 (텍스트 가이던스)
        guidance_embed = self.text_guidance_proj(guidance_vector)
        cond_embed = guidance_embed

        # 2. Decoder 입력 준비
        # `tgt`는 학습 가능한 위치 임베딩
        tgt = self.pos_embed.repeat(B, 1, 1)
        assert context_features.dim() == 3, f"context_features (B,S,D) 필요, got {context_features.shape}"
        if self.use_sensor_tokens:
            assert sensor_features is not None, "sensor_features가 필요합니다(use_sensor_tokens=True)"

        # 1) 컨디셔닝 벡터
        guidance_embed = self.text_guidance_proj(guidance_vector)
        cond_embed = guidance_embed  # (B,D)

        # 2) 메모리(비전+센서) 구성 + 포지셔널/타입 임베딩
        vision_mem = self.context_proj(context_features)              # (B,Sv,D)
        vision_mem = vision_mem + self.token_type_embed.weight[0].view(1,1,-1)

        if self.use_sensor_tokens:
            if sensor_features.dim() == 2:
                sensor_tok = self.sensor_proj(sensor_features).unsqueeze(1)  # (B,1,D)
            else:
                sensor_tok = self.sensor_proj(sensor_features)               # (B,Ss,D)
            sensor_tok = sensor_tok + self.token_type_embed.weight[1].view(1,1,-1)

            Sv = vision_mem.size(1)
            Ss = sensor_tok.size(1)
            vision_mem = vision_mem + self.mem_pos[:, :Sv]
            sensor_tok = sensor_tok + (self.mem_pos[:, :Ss] if Ss <= self.mem_pos.size(1) else 0.0)
            memory = torch.cat([vision_mem, sensor_tok], dim=1)
        else:
            Sv = vision_mem.size(1)
            memory = vision_mem + self.mem_pos[:, :Sv]

        # 3) 타깃 토큰(tgt) + 인과 마스크
        tgt = self.pos_embed.repeat(B, 1, 1) + cond_embed.unsqueeze(1)
        tgt_mask = None
        if self.causal_self_attn:
            H_ = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(H_, H_, device=tgt.device), diagonal=1).bool()

        # 4) 모듈식 디코더 스택
        x = tgt
        for i in range(self.num_decoder_layers):
            x = self.mod_layers[i](x, memory, cond_embed, tgt_mask=tgt_mask)
        decoded = x

        # 5) 델타 예측 + 안전 클램프
        delta_trans = self.trans_head(decoded)
        delta_rot = self.rot_head(decoded)
        delta_grip = self.grip_head(decoded)
        delta = torch.cat([delta_trans, delta_rot, delta_grip], dim=-1)
        delta = torch.clamp(delta, min=-self.max_delta, max=self.max_delta)

        pred_actions = z_chunk + delta
        return pred_actions, delta


# ==============================================================================
# 3. Deprecated 액션 전문가 (DiffusionActionExpert)
# ==============================================================================

class DiffusionSchedule:
    """
    확산 모델의 노이즈 스케줄 및 샘플링을 관리합니다.
    """
    def __init__(self, timesteps=100, schedule='cosine', device='cuda'):
        self.timesteps = timesteps
        self.device = device
        betas = cosine_beta_schedule(timesteps).to(device) if schedule == 'cosine' else None
        if betas is None: raise ValueError(f"알 수 없는 스케줄: {schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.to(self.device).detach())

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_recip * x_t - sqrt_recipm1 * eps

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_x0 = self.predict_x0_from_eps(x_t, t, eps_pred)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        model_mean = (
            self.sqrt_recip_alphas[t].view(-1, 1, 1) *
            (x_t - self.betas[t].view(-1, 1, 1) * eps_pred /
             self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1))
        )
        model_variance = self.posterior_variance[t].view(-1, 1, 1)
        model_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1, 1)
        return model_mean, model_variance, model_log_variance, pred_x0

class DiffusionActionExpert(nn.Module):
    """
    *** Deprecated: 모델 훈련 시 FlowMatchingActionExpert 또는 RegressionActionExpert를 사용하십시오. ***
    """
    def __init__(self,
                 vl_dim=3072,
                 sensor_dim=3072,
                 action_dim=7,
                 horizon=8,
                 hidden_dim=512,
                 timesteps=100,
                 fusion_strategy='concat',
                 nhead=8,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!! 경고: DiffusionActionExpert는 더 이상 사용되지 않으며, Flow Matching으로 대체되었습니다.  !!")
        print("!!             FlowMatchingActionExpert를 사용해주십시오.                        !!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.action_dim = action_dim
        self.horizon = horizon
        self.timesteps = timesteps
        self.fusion_strategy = fusion_strategy
        self.diffusion = DiffusionSchedule(timesteps=timesteps, schedule='cosine', device='cuda')
        self.time_embed = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        if fusion_strategy == 'concat':
            fused_dim = vl_dim + sensor_dim
            self.cond_proj = nn.Linear(fused_dim, hidden_dim)
        elif fusion_strategy == 'cross_attention':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
            self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'gated':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'none':
            self.cond_proj = nn.Linear(vl_dim, hidden_dim)
        else:
            raise ValueError(f"알 수 없는 융합 전략: {fusion_strategy}")
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.trans_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.rot_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.grip_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        print(f"✅ DiffusionActionExpert 초기화 완료:")
        print(f"   Timesteps: {timesteps}, 융합 전략: {fusion_strategy}")
        print(f"   행동 형태: (B, {horizon}, {action_dim})")

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor,
                vl_tokens: torch.Tensor, sensor_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H, A = noisy_actions.shape
        t_embed = self.timestep_embedding(timesteps, dtype=noisy_actions.dtype)
        t_embed = self.time_embed(t_embed)
        t_embed = t_embed.unsqueeze(1).expand(-1, H, -1)
        cond_embed = self._encode_condition(vl_tokens, sensor_features)
        cond_embed = cond_embed.unsqueeze(1).expand(-1, H, -1)
        action_embed = self.action_embed(noisy_actions)
        x = t_embed + cond_embed + action_embed
        x = self.temporal_encoder(x)
        eps_trans = self.trans_head(x)
        eps_rot = self.rot_head(x)
        eps_grip = self.grip_head(x)
        eps_pred = torch.cat([eps_trans, eps_rot, eps_grip], dim=-1)
        return eps_pred

    def _encode_condition(self, vl_tokens: torch.Tensor, sensor_features: Optional[torch.Tensor]) -> torch.Tensor:
        if self.fusion_strategy == 'concat' and sensor_features is not None:
            vl_pooled = vl_tokens.mean(dim=1)
            fused = torch.cat([vl_pooled, sensor_features], dim=-1)
            cond = self.cond_proj(fused)
        elif self.fusion_strategy == 'cross_attention' and sensor_features is not None:
            vl_feat = self.vl_proj(vl_tokens)
            sensor_feat = self.sensor_proj(sensor_features).unsqueeze(1)
            attn_out, _ = self.cross_attn(sensor_feat, vl_feat, vl_feat)
            cond = self.cond_proj(attn_out.squeeze(1))
        elif self.fusion_strategy == 'gated' and sensor_features is not None:
            vl_pooled = vl_tokens.mean(dim=1)
            vl_feat = self.vl_proj(vl_pooled)
            sensor_feat = self.sensor_proj(sensor_features)
            gate = self.gate(torch.cat([vl_feat, sensor_feat], dim=-1))
            fused = gate * vl_feat + (1 - gate) * sensor_feat
            cond = self.cond_proj(fused)
        else:
            vl_pooled = vl_tokens.mean(dim=1)
            cond = self.cond_proj(vl_pooled)
        return cond

    def timestep_embedding(self, timesteps: torch.Tensor, dim: int = 128, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb.to(dtype=dtype)

    @torch.no_grad()
    def sample(self, vl_tokens: torch.Tensor, sensor_features: Optional[torch.Tensor] = None,
               batch_size: int = 1, ddim_steps: Optional[int] = None) -> torch.Tensor:
        device = vl_tokens.device
        dtype = vl_tokens.dtype
        H, A = self.horizon, self.action_dim
        x = torch.randn(batch_size, H, A, device=device, dtype=dtype)
        if ddim_steps is not None:
            return self._ddim_sample(x, vl_tokens, sensor_features, ddim_steps)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps_pred = self.forward(x, t_batch, vl_tokens, sensor_features)
            mean, variance, log_variance, _ = self.diffusion.p_mean_variance(x, t_batch, eps_pred)
            noise = torch.randn_like(x) if t > 0 else 0.0
            x = mean + torch.sqrt(variance) * noise
        return x

    @torch.no_grad()
    def _ddim_sample(self, x: torch.Tensor, vl_tokens: torch.Tensor, sensor_features: Optional[torch.Tensor], ddim_steps: int) -> torch.Tensor:
        device = x.device
        batch_size = x.shape[0]
        step_size = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, step_size))[:ddim_steps]
        timesteps = list(reversed(timesteps))
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps_pred = self.forward(x, t_batch, vl_tokens, sensor_features)
            pred_x0 = self.diffusion.predict_x0_from_eps(x, t_batch, eps_pred)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_next = self.diffusion.alphas_cumprod[t_next]
            else:
                alpha_next = torch.tensor(1.0, device=device)
            alpha_t = self.diffusion.alphas_cumprod[t]
            sigma_t = 0.0
            x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next - sigma_t**2) * eps_pred
        return x
