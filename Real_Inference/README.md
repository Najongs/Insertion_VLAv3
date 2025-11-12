# Real-time VLA Inference

실시간 VLA 추론 시스템 with 성능 모니터링

## 🚀 시스템 구성

```
[보드 PC] Camera Sender
  └─> 5 views @ 640x360, 5Hz (ZMQ PUSH port 5555)

[로봇 PC] Robot Sender
  └─> 6 joints + 6 poses, 100Hz (ZMQ PUB port 5556)

[센서 PC] Sensor Sender (C++)
  └─> Force + OCT A-scan, 650Hz (UDP port 9999)

[서버 PC] Async Inference Receiver
  ├─ Image Buffer (5 views)
  ├─ Sensor Buffer (65 samples, 100ms window)
  ├─ Robot State Buffer (65 samples, 100ms window)
  └─> VLA Model (flow_matching or regression)
       └─> Actions @ 10Hz
```

## 📋 실행 순서

### 1. Camera Sender 실행 (보드 PC)
```bash
cd /home/najo/NAS/VLA/Insertion_VLAv2
python Real_Inference/Optimized_Camera_sender.py
```

### 2. Robot Sender 실행 (로봇 PC, IP: 10.130.41.111)
```bash
cd /home/najo/NAS/VLA/Insertion_VLAv2
python Real_Inference/Robot_sender.py --robot on
```

### 3. Sensor Sender 실행 (센서 PC)
```bash
# C++ 코드 실행 (UDP port 9999로 전송)
```

### 4. Inference Receiver 실행 (서버 PC)

#### Flow Matching 모델 (권장)
```bash
python Real_Inference/Async_inference_receiver.py \
    --checkpoint checkpoints/flow_matching_best.pt \
    --model-type flow_matching \
    --flow-steps 10
```

#### Regression 모델
```bash
python Real_Inference/Async_inference_receiver.py \
    --checkpoint checkpoints/regression_best.pt \
    --model-type regression
```

#### 데이터 저장 모드
```bash
python Real_Inference/Async_inference_receiver.py \
    --checkpoint checkpoints/regression_best.pt \
    --model-type regression \
    --save-data
```

## 📊 성능 결과

종료 시 (Ctrl+C) 다음 파일들이 `Real_Inference/async_inference_YYYYMMDD_HHMMSS/` 폴더에 자동 저장됩니다:

### 1. **performance_summary_YYYYMMDD_HHMMSS.json**
전체 성능 요약 통계
```json
{
  "total_actions": 100,
  "elapsed_time_sec": 10.5,
  "average_fps": 9.52,
  "vl_encoding": {
    "mean_ms": 380.5,
    "p50_ms": 360.0,
    "p95_ms": 420.0,
    "p99_ms": 450.0
  },
  "action_prediction": {
    "mean_ms": 18.3,
    "p50_ms": 15.0,
    "p95_ms": 25.0,
    "p99_ms": 130.0
  }
}
```

### 2. **performance_timings_YYYYMMDD_HHMMSS.jsonl**
모든 추론의 상세 타이밍 데이터 (JSON Lines 형식 - 실시간 자동 저장!)

**JSON Lines 형식**: 각 줄이 하나의 JSON 객체 (10개 레코드마다 자동 저장)

```jsonl
{"timestamp": 1730..., "action_id": 1, "vl_update_number": 1, "sensor_encoding_time": 2.1, "robot_encoding_time": 1.8, "action_prediction_time": 15.3, "total_time": 235.0}
{"timestamp": 1730..., "action_id": 2, "vl_update_number": 1, "sensor_encoding_time": 2.0, "robot_encoding_time": 1.9, "action_prediction_time": 14.8, "total_time": 15.0}
{"timestamp": 1730..., "action_id": 3, "vl_update_number": 1, "sensor_encoding_time": 1.9, "robot_encoding_time": 1.7, "action_prediction_time": 14.5, "total_time": 16.2}
...
```

**장점:**
- ⚡ 실시간 자동 저장 (종료 시 대기 불필요!)
- 💾 10개 레코드마다 파일에 추가 저장
- 🔒 Ctrl+C 중단해도 데이터 손실 없음

### 3. **inference_results_YYYYMMDD_HHMMSS.json**
모델 출력 (actions, deltas)

## 🔍 성능 분석 예시

### 방법 0: 빠른 분석 스크립트 (권장)
```bash
python Real_Inference/analyze_performance.py \
    async_inference_20251104_234931/performance_timings_20251104_234931.jsonl
```

> **참고**: analyze_performance.py는 JSON과 JSON Lines 형식 모두 지원합니다

출력 예시:
```
================================================================================
Performance Analysis: performance_timings_20251104_234931.json
================================================================================
Total Actions: 100
Duration: 10.5s
Average FPS: 9.52 Hz

Action Prediction Time:
  Mean:  18.30ms
  P95:   25.00ms
  P99:   130.00ms

✅ EXCELLENT
Comment: Meeting 10Hz target comfortably
Mean Action Time: 18.3ms
P95 Total Time: 28.5ms
Target: 100ms per inference (10Hz)
================================================================================
```

### 방법 1: JSON Lines 직접 분석 (빠름)
```python
import json
import numpy as np

# JSON Lines 로드
timings = []
with open('async_inference_20251104_234931/performance_timings_20251104_234931.jsonl') as f:
    for line in f:
        if line.strip():
            timings.append(json.loads(line))

# 통계 계산
action_times = [t['action_prediction_time'] for t in timings]
print(f"Mean: {np.mean(action_times):.1f}ms")
print(f"P95: {np.percentile(action_times, 95):.1f}ms")
print(f"P99: {np.percentile(action_times, 99):.1f}ms")
```

### 방법 2: pandas 사용 (상세 분석)
```python
import pandas as pd
import matplotlib.pyplot as plt
import json

# JSON Lines 로드 후 DataFrame으로 변환
timings = []
with open('async_inference_20251104_234931/performance_timings_20251104_234931.jsonl') as f:
    for line in f:
        if line.strip():
            timings.append(json.loads(line))
df = pd.DataFrame(timings)

# 통계 확인
print(df.describe())

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(df['action_prediction_time'], bins=50)
axes[0, 0].set_title('Action Prediction Time (ms)')

axes[0, 1].hist(df['sensor_encoding_time'], bins=50)
axes[0, 1].set_title('Sensor Encoding Time (ms)')

axes[1, 0].hist(df['robot_encoding_time'], bins=50)
axes[1, 0].set_title('Robot State Encoding Time (ms)')

axes[1, 1].hist(df['total_time'], bins=50)
axes[1, 1].set_title('Total Inference Time (ms)')

plt.tight_layout()
plt.savefig('performance_analysis.png')
```

## ⚙️ CLI 옵션

```bash
python Real_Inference/Async_inference_receiver.py --help
```

- `--checkpoint PATH`: 모델 체크포인트 경로 (필수)
- `--model-type {flow_matching,regression}`: 모델 타입 (기본: flow_matching)
- `--flow-steps N`: Flow matching ODE 스텝 (기본: 10)
- `--vl-reuse N`: VL features 재사용 횟수 (기본: 4)
- `--save-data`: 이미지/센서/로봇 데이터 저장

## 📈 예상 성능 (RTX 3090 기준)

### ⚡ 병렬 VL 인코딩 활성화됨!
**5개 이미지 병렬 처리로 2-3배 빠른 VL 업데이트**

### Flow Matching (10 steps) with Parallel Encoding
- **VL Encoding**: ~150-200ms (5-6Hz) ⚡ (이전: 360ms)
- **Sensor Encoding**: ~2ms
- **Robot State Encoding**: ~2ms
- **Action Prediction**: ~15ms (p95: ~25ms)
- **Total**: ~20ms @ 10Hz

### Regression with Parallel Encoding
- **VL Encoding**: ~150-200ms (5-6Hz) ⚡ (이전: 360ms)
- **Sensor Encoding**: ~2ms
- **Robot State Encoding**: ~2ms
- **Action Prediction**: ~10ms (p95: ~20ms)
- **Total**: ~15ms @ 10Hz

## 🐛 문제 해결

### Robot buffer가 채워지지 않음
```
[WAIT] Robot: False (0/65)
```
**해결**: Robot_sender.py가 `10.130.41.111:5556`에서 실행 중인지 확인

### Sensor buffer가 채워지지 않음
```
[WAIT] Sensor: False (0/65)
```
**해결**: Sensor sender (C++)가 UDP port 9999로 전송 중인지 확인

### VL 업데이트가 느림
```
VL Updates: 10 | VL avg: 800ms
```
**해결**:
1. 이미지 해상도 확인 (640x360 권장)
2. GPU 사용률 확인 (`nvidia-smi`)
3. `--vl-reuse` 값 증가 (예: 8)

## 📁 출력 파일 구조

```
async_inference_20251104_234931/
├── performance_summary_20251104_234931.json    # 성능 통계 (요약)
├── performance_timings_20251104_234931.jsonl   # 상세 타이밍 (실시간 자동 저장!)
├── inference_results_20251104_234931.json      # 모델 출력
├── robot_state_20251104_234931.csv            # 로봇 데이터 (--save-data)
├── sensor_data_20251104_234931.npz            # 센서 데이터 (--save-data)
└── images/                                     # 이미지 (--save-data)
    ├── View1/
    ├── View2/
    ├── View3/
    ├── View4/
    └── View5/
```

> **참고**:
> - 상세 타이밍 데이터는 JSON Lines (`.jsonl`) 형식으로 실시간 저장됩니다
> - 10개 레코드마다 자동으로 파일에 추가 저장되어 종료 시 대기 불필요!
> - Ctrl+C로 중단해도 데이터 손실 없음

## 🎯 최적화 팁

1. **VL 재사용 조정**: `--vl-reuse 8`로 VL 업데이트 빈도 감소
2. **Flow 스텝 조정**: `--flow-steps 5`로 속도 향상 (정확도 trade-off)
3. **이미지 해상도**: 640x360 유지 (Camera sender에서 설정)
4. **GPU 전용**: 추론 중 다른 GPU 작업 최소화

---

**문의**: 문제 발생 시 로그 전체를 공유해주세요.
