# VLA Model Evaluation System

포괄적인 VLA 모델 평가 시스템입니다. 학습된 Flow Matching 모델의 성능을 다각도로 분석하고 시각화합니다.

## 📋 평가 메트릭

### 1. **Position Metrics (위치 정확도)**
- **RMSE** (Root Mean Square Error): 전체 위치 오차의 제곱근 평균
- **MAE** (Mean Absolute Error): 평균 절대 오차
- **Max Error**: 최대 오차
- **Success Rate**: threshold 이하 오차의 비율 (기본: 5mm)

### 2. **Rotation Metrics (회전 정확도)**
- **RMSE**: 회전 오차의 제곱근 평균 (degrees)
- **MAE**: 평균 절대 회전 오차
- **Max Error**: 최대 회전 오차

### 3. **Gripper Metrics (그리퍼 정확도)**
- **Accuracy**: 그리퍼 open/close 예측 정확도 (%)

### 4. **Per-Dimension Analysis**
- dx, dy, dz, rx, ry, rz, gripper 각 차원별 RMSE/MAE

## 🚀 사용 방법

### 방법 1: 통합 스크립트 사용 (권장)

```bash
bash RUN_EVALUATION.sh
```

이 스크립트는 자동으로:
1. 여러 에피소드에서 모델 평가
2. 통합 메트릭 계산 및 JSON 저장
3. 모든 시각화 플롯 생성
4. wandb에 결과 로깅 (옵션)

### 방법 2: 개별 스크립트 실행

#### Step 1: 평가 실행
```bash
python EVAL_FlowMatching.py \
    --checkpoint checkpoints/flow_matching_best.pt \
    --dataset-paths "/home/najo/NAS/VLA/dataset/New_dataset6/*_point" \
    --output-dir evaluation_results/my_eval \
    --batch-size 4 \
    --threshold-mm 5.0 \
    --sensor-hidden-dim 512 \
    --sensor-transformer-dim 1024 \
    --use-cache \
    --cache-only-mode \
    --wandb-project "QwenVLA-Evaluation"
```

#### Step 2: 시각화 생성
```bash
python evaluation_results/plot_evaluation_results.py \
    --results-json evaluation_results/my_eval/evaluation_results_flow_matching_best.json \
    --output-dir evaluation_results/my_eval/plots
```

### 방법 3: 단일 에피소드 평가 (기존 방식)

```bash
python evaluate_flowmatching_episode.py \
    --episode-path /path/to/episode \
    --checkpoint checkpoints/flow_matching_best.pt \
    --output evaluation_results/single_episode_eval.json \
    --sensor-hidden-dim 512 \
    --batch-size 4
```

## 📊 생성되는 시각화 파일

평가 완료 후 `plots/` 디렉토리에 다음 파일들이 생성됩니다:

1. **`*_summary_report.png`**: 전체 평가 요약 (메트릭 + 설정 정보)
2. **`*_error_distribution.png`**: 각 차원별 오차 분포 히스토그램
3. **`*_per_dim_rmse.png`**: 차원별 RMSE 바 차트
4. **`*_3d_trajectories.png`**: 3D 위치 궤적 비교 (GT vs Predicted)
5. **`*_temporal_error.png`**: Horizon에 따른 시간별 오차 분석
6. **`*_per_episode_metrics.png`**: 에피소드별 성능 비교

## 📁 출력 파일 구조

```
evaluation_results/
├── evaluation_results_flow_matching_best.json  # 전체 평가 결과 (JSON)
└── plots/
    ├── evaluation_results_flow_matching_best_summary_report.png
    ├── evaluation_results_flow_matching_best_error_distribution.png
    ├── evaluation_results_flow_matching_best_per_dim_rmse.png
    ├── evaluation_results_flow_matching_best_3d_trajectories.png
    ├── evaluation_results_flow_matching_best_temporal_error.png
    └── evaluation_results_flow_matching_best_per_episode_metrics.png
```

## 🎯 주요 기능

### 1. **여러 에피소드 동시 평가**
- Glob 패턴 지원으로 여러 에피소드 자동 탐색
- Per-episode + Overall 메트릭 제공

### 2. **Validation Split 지원**
```bash
# 전체 데이터셋의 마지막 10%를 검증 세트로 사용
python EVAL_FlowMatching.py \
    --dataset-paths "/path/to/dataset/*" \
    --val-split 0.1
```

### 3. **샘플링 평가**
```bash
# 전체 에피소드 중 랜덤하게 10개만 평가
python EVAL_FlowMatching.py \
    --dataset-paths "/path/to/dataset/*" \
    --sample-episodes 10
```

### 4. **wandb 로깅**
```bash
python EVAL_FlowMatching.py \
    --checkpoint checkpoints/flow_matching_best.pt \
    --dataset-paths "/path/to/data" \
    --wandb-project "QwenVLA-Evaluation" \
    --wandb-name "my_eval_run"
```

## 📈 평가 결과 해석

### Success Rate (성공률)
- **> 90%**: 매우 좋은 성능
- **70-90%**: 양호한 성능
- **< 70%**: 개선 필요

### Position RMSE (mm)
- **< 3mm**: 매우 정밀
- **3-10mm**: 정밀한 작업 가능
- **> 10mm**: 개선 필요

### Rotation RMSE (degrees)
- **< 5°**: 매우 정밀
- **5-15°**: 정밀한 작업 가능
- **> 15°**: 개선 필요

## 🔧 고급 옵션

### 커스텀 Success Threshold
```bash
python EVAL_FlowMatching.py \
    --threshold-mm 3.0  # 3mm 이하를 성공으로 간주
```

### 센서/로봇 상태 비활성화
```bash
python EVAL_FlowMatching.py \
    --disable-sensor        # 센서 인코더 비활성화
    --disable-robot-state   # 로봇 상태 인코더 비활성화
```

### 캐시 없이 평가 (실시간 VLM 인코딩)
```bash
python EVAL_FlowMatching.py \
    --no-cache  # VL 캐시 사용 안 함 (느림)
```

## 📝 JSON 결과 구조

```json
{
  "checkpoint": "path/to/checkpoint.pt",
  "checkpoint_meta": {
    "epoch": 50,
    "val_loss": 0.0123,
    "best_val_loss": 0.0115
  },
  "model_info": {
    "total_params": 3452817408,
    "trainable_params": 152817408,
    "non_trainable_params": 3300000000,
    "model_size_mb": 13172.45,
    "model_size_gb": 12.86,
    "component_params": {
      "vl_model": {
        "total": 3300000000,
        "trainable": 0,
        "size_mb": 12595.37
      },
      "sensor_encoder": {
        "total": 75280384,
        "trainable": 75280384,
        "size_mb": 287.23
      },
      "robot_state_encoder": {
        "total": 25125888,
        "trainable": 25125888,
        "size_mb": 95.87
      },
      "flow_decoder": {
        "total": 52411136,
        "trainable": 52411136,
        "size_mb": 193.98
      }
    }
  },
  "evaluation_config": {
    "num_episodes": 3,
    "threshold_mm": 5.0,
    "horizon": 8,
    "action_dim": 7,
    "view_indices": null,
    "disable_sensor": false,
    "disable_robot_state": false
  },
  "overall_metrics": {
    "position": {
      "rmse_mm": 4.567,
      "mae_mm": 3.234,
      "success_rate": 0.856
    },
    "rotation": {...},
    "gripper": {...},
    "per_dimension": {...}
  },
  "overall_timing": {
    "total_inference_time_sec": 45.23,
    "avg_time_per_sample_ms": 52.3,
    "total_samples": 865
  },
  "episodes": [
    {
      "episode_name": "data_collection_20251117_232047",
      "num_samples": 145,
      "metrics": {...},
      "timing": {
        "total_inference_time_sec": 7.56,
        "avg_time_per_sample_ms": 52.1,
        "total_samples": 145
      },
      "samples": [...]
    }
  ]
}
```

### 모델 정보 (Model Info)
평가 시작 시 자동으로 출력되는 모델 정보:
- **Total Parameters**: 전체 파라미터 수 (M 단위)
- **Trainable Parameters**: 학습 가능한 파라미터 수
- **Non-trainable Parameters**: Frozen된 파라미터 수 (VL 모델 등)
- **Model Size**: 모델 전체 크기 (MB/GB)
- **Component Breakdown**: VL 모델, Sensor Encoder, Robot State Encoder, Flow Decoder 각각의 파라미터 수 및 크기

### 추론 시간 (Inference Timing)
각 평가마다 자동으로 측정되는 추론 시간 정보:
- **Total Inference Time**: 전체 추론 소요 시간 (초)
- **Avg Time per Sample**: 샘플당 평균 추론 시간 (ms)
- **Total Samples**: 평가한 총 샘플 수

## 🔬 Ablation Study (성분 제거 실험)

학습된 모델에서 각 컴포넌트(Sensor Encoder, Robot State Encoder)의 기여도를 분석하기 위한 ablation study를 자동으로 실행합니다.

### 통합 Ablation Study 실행

```bash
bash RUN_ABLATION_STUDY.sh
```

이 스크립트는 자동으로 4가지 설정을 평가합니다:
1. **Full Model** (baseline): 모든 컴포넌트 사용
2. **w/o Sensor**: Sensor Encoder 제거
3. **w/o Robot State**: Robot State Encoder 제거
4. **w/o Both**: 두 인코더 모두 제거 (VL 모델만 사용)

### 개별 Ablation 평가 실행

원하는 설정만 평가하려면:

```bash
# Sensor Encoder 없이 평가
python EVAL_FlowMatching.py \
    --checkpoint checkpoints/flow_matching_best.pt \
    --dataset-paths "/path/to/dataset/*" \
    --output-dir evaluation_results/ablation_wo_sensor \
    --disable-sensor

# Robot State Encoder 없이 평가
python EVAL_FlowMatching.py \
    --checkpoint checkpoints/flow_matching_best.pt \
    --dataset-paths "/path/to/dataset/*" \
    --output-dir evaluation_results/ablation_wo_robot \
    --disable-robot-state

# 두 인코더 모두 없이 평가 (VL only)
python EVAL_FlowMatching.py \
    --checkpoint checkpoints/flow_matching_best.pt \
    --dataset-paths "/path/to/dataset/*" \
    --output-dir evaluation_results/ablation_wo_both \
    --disable-sensor \
    --disable-robot-state
```

### Ablation 결과 비교 시각화

```bash
python evaluation_results/plot_ablation_comparison.py \
    --full-model evaluation_results/ablation_study/full_model/evaluation_results_*.json \
    --wo-sensor evaluation_results/ablation_study/wo_sensor/evaluation_results_*.json \
    --wo-robot-state evaluation_results/ablation_study/wo_robot_state/evaluation_results_*.json \
    --wo-both evaluation_results/ablation_study/wo_both/evaluation_results_*.json \
    --output-dir evaluation_results/ablation_study/comparison_plots
```

### 생성되는 Ablation 비교 플롯

1. **`ablation_overall_metrics_comparison.png`**:
   - Position RMSE, Rotation RMSE, Success Rate, Gripper Accuracy 비교
   - 각 설정별 절대 성능 표시

2. **`ablation_per_dimension_comparison.png`**:
   - dx, dy, dz, rx, ry, rz, gripper 각 차원별 RMSE 비교
   - 어느 차원에서 성능 차이가 큰지 분석

3. **`ablation_relative_performance.png`**:
   - Full model 대비 상대적 성능 변화 (%)
   - 각 컴포넌트 제거 시 성능 저하 정도 표시

4. **`ablation_summary_table.png`**:
   - 모든 주요 메트릭을 포함한 요약 테이블
   - 각 설정별 상세 수치 비교

### Ablation 결과 해석 가이드

#### Sensor Encoder의 중요도
- **w/o Sensor**와 **Full Model** 비교
- Position RMSE 증가 > 5mm: Sensor가 위치 제어에 중요
- Gripper Accuracy 감소 > 10%: 접촉 감지에 센서 필수

#### Robot State Encoder의 중요도
- **w/o Robot State**와 **Full Model** 비교
- Rotation RMSE 증가 > 5°: Robot state가 자세 제어에 중요
- Success Rate 감소 > 20%: 동적 정보가 성공률에 필수

#### Vision-Language 모델만의 성능
- **w/o Both** 결과 분석
- VL 모델이 얼마나 많은 정보를 캡처하는지 확인
- 센서/로봇 정보 없이도 reasonable한 성능이 나오는지 검증

### 예상 결과 패턴

일반적으로 다음과 같은 패턴이 예상됩니다:

```
Position RMSE (mm):
Full Model < w/o Sensor < w/o Robot State < w/o Both

Success Rate (%):
Full Model > w/o Robot State > w/o Sensor > w/o Both

Gripper Accuracy (%):
Full Model > w/o Robot State > w/o Sensor > w/o Both
```

만약 예상과 다른 결과가 나온다면:
- **w/o Sensor가 Full보다 좋음**: Sensor 인코더가 과적합되었거나 노이즈 추가
- **w/o Robot State가 Full보다 좋음**: Robot state 인코더가 불필요하거나 잘못 학습됨
- **차이가 거의 없음**: VL 모델이 이미 충분한 정보를 제공하거나, 인코더 학습 부족

## 🎥 View Ablation Study (카메라 뷰 제거 실험)

카메라 뷰의 개수와 특정 뷰의 기여도를 분석하기 위한 ablation study입니다. 성능과 추론 시간의 trade-off를 분석하여 실제 배포 시 최적의 카메라 설정을 찾을 수 있습니다.

### 통합 View Ablation Study 실행

```bash
bash RUN_VIEW_ABLATION_STUDY.sh
```

이 스크립트는 자동으로 9가지 설정을 평가합니다:
1. **All views (5개)**: 모든 카메라 뷰 사용 (baseline)
2. **4 views (0-3)**: View 4 제거
3. **3 views (0-2)**: View 3,4 제거
4. **2 views (0-1)**: View 2,3,4 제거
5-9. **Single views**: 각 뷰를 개별적으로 사용 (View 0, 1, 2, 3, 4)

### 개별 View 평가 실행

특정 뷰 조합만 평가하려면:

```bash
# 특정 뷰만 사용 (예: View 0, 1만)
python EVAL_FlowMatching.py \
    --checkpoint checkpoints/flow_matching_best.pt \
    --dataset-paths "/path/to/dataset/*" \
    --output-dir evaluation_results/view_ablation_views_01 \
    --view-indices 0 1

# 단일 뷰만 사용 (예: View 2만)
python EVAL_FlowMatching.py \
    --checkpoint checkpoints/flow_matching_best.pt \
    --dataset-paths "/path/to/dataset/*" \
    --output-dir evaluation_results/view_ablation_view_2 \
    --view-indices 2
```

### 생성되는 View Ablation 비교 플롯

1. **`view_ablation_performance_vs_num_views.png`**:
   - 뷰 개수에 따른 성능 변화 (line plot)
   - Position RMSE, Rotation RMSE, Success Rate, Inference Time 비교

2. **`view_ablation_single_view_comparison.png`**:
   - 각 single view (0-4) 간 성능 비교 (bar chart)
   - 어느 뷰가 가장 중요한 정보를 제공하는지 분석

3. **`view_ablation_performance_inference_tradeoff.png`**:
   - 성능 vs 추론 시간 trade-off 분석 (scatter plot)
   - Pareto frontier를 통해 최적 설정 찾기

4. **`view_ablation_trajectory_comparison.png`**:
   - **3D 트레젝토리 시각화** (All views vs 2 views vs Single view)
   - GT와 예측 궤적의 시각적 비교

5. **`view_ablation_summary_table.png`**:
   - 모든 설정의 상세 메트릭 및 추론 시간 테이블

### View Ablation 결과 해석 가이드

#### 뷰 개수의 영향
- **5 views → 4 views**: 성능 하락 < 5% → 하나의 뷰는 redundant
- **4 views → 3 views**: 성능 하락 < 10% → 3개 뷰로도 충분
- **3 views → 2 views**: 성능 하락 > 15% → 최소 3개 뷰 필요
- **2 views → 1 view**: 성능 하락 > 30% → Multi-view가 필수

#### Single View 분석
각 뷰의 성능을 비교하여:
- **가장 좋은 single view**: 가장 중요한 시점
- **가장 나쁜 single view**: 제거 가능한 뷰
- **뷰 간 성능 차이**: 시점의 중요도 차이

#### Inference Time Trade-off
- **All views (5개)**: 최고 성능, 최대 추론 시간
- **2-3 views**: 균형잡힌 성능-속도 trade-off
- **Single view**: 최소 추론 시간, 성능 저하

실제 배포 시:
- **실시간 제어 필요**: 2-3 views 사용 (빠른 추론)
- **최고 정확도 필요**: All views 사용
- **리소스 제약**: Single view (best performing view)

#### 3D Trajectory 분석
트레젝토리 시각화를 통해:
- **All views**: GT에 가장 근접
- **Reduced views**: 어느 방향에서 오차가 발생하는지 확인
- **Single view**: 특정 축(x/y/z)에서 큰 편차 발생

### 예상 결과 패턴

일반적으로:

```
Position RMSE (mm):
All views < 4 views < 3 views < 2 views < Single view

Inference Time (ms/sample):
All views > 4 views > 3 views > 2 views > Single view

Success Rate (%):
All views > 4 views > 3 views > 2 views > Single view
```

최적 설정 예시:
- **Production (실시간)**: 2-3 views, ~50ms/sample, 85%+ success rate
- **Research (최고 정확도)**: All views, ~100ms/sample, 90%+ success rate

## 🐛 문제 해결

### "No episodes found" 오류
- 데이터셋 경로가 올바른지 확인
- metadata.json 파일이 에피소드 디렉토리에 있는지 확인

### "Cache not found" 오류
- `--no-cache` 옵션 사용 (느리지만 작동)
- 또는 VL 캐시를 먼저 생성 (TOTAL_TRAIN.sh STEP 1)

### OOM (Out of Memory) 오류
- `--batch-size` 줄이기 (4 → 2 → 1)
- `--cache-only-mode` 사용으로 VLM 메모리 절약

## 📞 참고

자세한 내용은 상위 디렉토리의 메인 README.md를 참조하세요.
