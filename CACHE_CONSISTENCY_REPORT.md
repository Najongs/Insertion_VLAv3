# VL Cache 일관성 분석 보고서

## 📋 분석 대상
- `TRAIN_FlowMatching.py`
- `TRAIN_Regression.py`
- `TOTAL_TRAIN.sh`
- `Make_VL_cache.py`
- `unified_dataset.py`

---

## ❌ 문제점 1: 캐시 경로 불일치

### 현재 상태

| 파일 | 기본 캐시 경로 |
|------|---------------|
| **TRAIN_FlowMatching.py** | `/home/najo/NAS/VLA/dataset/cache/qwen_vl_features` |
| **TRAIN_Regression.py** | `/home/najo/NAS/VLA/dataset/cache/qwen_vl_features` |
| **TOTAL_TRAIN.sh** | `CACHE_ROOT="/home/najo/NAS/VLA/dataset/cache"` |
| **unified_dataset.py** | `/home/najo/NAS/VLA/dataset/cache/qwen_vl_features` |

### 문제
- `TOTAL_TRAIN.sh`의 `CACHE_ROOT`는 `/home/najo/NAS/VLA/dataset/cache`로 설정
- 실제 캐시 저장 경로는 `/home/najo/NAS/VLA/dataset/cache/qwen_vl_features`
- **`TOTAL_TRAIN.sh`에 VL 캐시 생성 명령이 주석 처리되어 있음!**

### 영향
- 주석 해제 시 경로 불일치로 인해 캐시를 찾지 못할 수 있음

---

## ❌ 문제점 2: TOTAL_TRAIN.sh의 VL 캐시 생성 섹션 주석 처리

### 현재 상태 (Line 173-186)
```bash
# echo ""
# echo "=============== 0. VL CACHE BUILDING ==============="
# echo "Building VL feature cache for faster training..."
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_FlowMatching.py \
#     --mode cache \
#     --dataset_paths "${DATASET_PATHS[@]}" \
#     --batch_size $MAIN_BATCH_SIZE \
#     --num_workers 8 \
#     --image_resize_height $IMG_HEIGHT \
#     --image_resize_width $IMG_WIDTH \
#     --cache_loader_only \
#     --cache_root $QWEN_CACHE_ROOT
```

### 문제
1. **변수 미정의**: `$MAIN_BATCH_SIZE`, `$IMG_HEIGHT`, `$IMG_WIDTH`, `$QWEN_CACHE_ROOT` 변수가 정의되지 않음
2. **실행 불가**: 주석 해제해도 변수가 없어 실행 실패

---

## ❌ 문제점 3: FlowMatching vs Regression의 캐시 빌드 방식 차이

### TRAIN_FlowMatching.py (Line 706-727)
```python
# QwenVLAUnified 모델을 직접 생성
model = QwenVLAUnified(
    model_type='flow_matching',
    vl_model_name=vl_model_name,
    sensor_enabled=False,
    external_cache_root=args.cache_root,
)
model = model.to(device)
model.cache_dir = cache_dir  # 수동으로 cache_dir 할당

build_vl_cache_distributed_optimized(
    model=model,
    dataset=train_loader.dataset,
    device=device,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)
```

### TRAIN_Regression.py (Line 928-997)
```python
# Processor와 VL 모델을 수동으로 로드
processor = AutoProcessor.from_pretrained(vl_model_name)
vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)

# DummyVLA 래퍼 생성
class DummyVLA:
    def __init__(self, vl_model, processor, cache_dir: Path):
        self.vl_model = vl_model
        self.processor = processor
        self.cache_dir = cache_dir
        self._cache_path = QwenVLAUnified._cache_path.__get__(self)
        ...

dummy_model = DummyVLA(vl_model, processor, cache_dir)
build_vl_cache_distributed_optimized(
    dummy_model,
    full_dataset,
    device=device,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    prefetch_factor=4,
)
```

### 문제
- **두 스크립트가 다른 방식으로 캐시 빌드**
- FlowMatching: 정식 QwenVLAUnified 모델 사용
- Regression: DummyVLA 래퍼 사용 (QwenVLAUnified 메서드 참조)
- **일관성 부족**: DummyVLA가 QwenVLAUnified의 내부 메서드를 직접 참조하는 것은 취약함

---

## ❌ 문제점 4: Make_VL_cache.py의 모델 요구사항

### Make_VL_cache.py (Line 45-47)
```python
model 요구사항:
  - model.vl_model, model.processor 필요
  - (선택) model.cache_dir 있으면 사용, 없으면 cache_dir_fallback 사용
```

### 실제 사용
- **FlowMatching**: ✅ model.vl_model, model.processor, model.cache_dir 모두 존재
- **Regression**: ⚠️ DummyVLA로 간접 제공 (model.vl_model, model.processor, model.cache_dir)

### 문제
- DummyVLA는 QwenVLAUnified의 private 메서드를 직접 참조
- 향후 QwenVLAUnified 리팩토링 시 DummyVLA가 깨질 수 있음

---

## ✅ 올바른 작동 확인

### Make_VL_cache.py의 캐시 저장 로직 (Line 209-214)
```python
for j, item in enumerate(sub_items):
    pooled_single = pooled_batch[j:j+1]
    cache_mgr.save_cache(
        dataset_name=item["dataset_name"],
        vlm_idx=item["vlm_idx"],
        prompt_hash=item["prompt_hash"],
        vl_features=pooled_single
    )
```

### VLACacheManager.save_cache() (vla_cache_manager.py:90-106)
```python
def save_cache(self, dataset_name: str, vlm_idx: int, prompt_hash: str, vl_features: torch.Tensor):
    cache_path = self.get_cache_path(dataset_name, vlm_idx, prompt_hash)
    # cache_path = {cache_dir}/{prompt_hash}/{dataset_name}_vlm{vlm_idx}.pt
    self._atomic_save(vl_features.detach().to("cpu", dtype=torch.float16), cache_path)
    self._enforce_cache_limit()
```

### 경로 생성 로직 확인
- `cache_dir` → `/home/najo/NAS/VLA/dataset/cache/qwen_vl_features`
- `prompt_hash` → `"abc12345"` (예시)
- `dataset_name` → `"data_collection_20251108_054442"`
- `vlm_idx` → `0`

**최종 캐시 파일 경로**:
```
/home/najo/NAS/VLA/dataset/cache/qwen_vl_features/abc12345/data_collection_20251108_054442_vlm0.pt
```

✅ **경로 생성 로직은 올바름**

---

## 📊 권장 사항

### 1. TOTAL_TRAIN.sh 수정 필요

#### 변수 정의 추가
```bash
# VL Cache 생성을 위한 변수 정의
MAIN_BATCH_SIZE=8
IMG_HEIGHT=360
IMG_WIDTH=640
QWEN_CACHE_ROOT="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features"
```

#### VL 캐시 생성 섹션 활성화 (주석 제거)
```bash
echo ""
echo "=============== 0. VL CACHE BUILDING ==============="
echo "Building VL feature cache for faster training..."
torchrun --nproc_per_node=$NUM_GPUS TRAIN_FlowMatching.py \
    --mode cache \
    --dataset_paths "${DATASET_PATHS[@]}" \
    --batch_size $MAIN_BATCH_SIZE \
    --num_workers 8 \
    --image_resize_height $IMG_HEIGHT \
    --image_resize_width $IMG_WIDTH \
    --cache_root $QWEN_CACHE_ROOT
echo "=============== VL CACHE BUILDING COMPLETE ==============="
echo ""
```

**주의**: `--cache_loader_only` 플래그 제거 (TRAIN_FlowMatching.py가 자동으로 처리)

---

### 2. TRAIN_Regression.py의 DummyVLA 제거 (선택 사항)

FlowMatching과 동일한 방식으로 통일:

```python
if args.mode == 'cache':
    # FlowMatching과 동일한 방식
    train_loader, _ = build_dataloaders(
        args, rank, world_size,
        use_cache=True,
        cache_build_only=True,
    )

    model = QwenVLAUnified(
        model_type='regression',
        vl_model_name=vl_model_name,
        sensor_enabled=False,
        external_cache_root=args.cache_root,
        ...
    )
    model = model.to(device)
    model.cache_dir = Path(args.cache_root)

    build_vl_cache_distributed_optimized(
        model=model,
        dataset=train_loader.dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
```

---

### 3. 캐시 경로 일관성 확인

모든 파일에서 동일한 경로 사용:
```
/home/najo/NAS/VLA/dataset/cache/qwen_vl_features
```

---

## 🎯 요약

| 항목 | 상태 | 비고 |
|------|------|------|
| 캐시 저장 로직 | ✅ 정상 | VLACacheManager가 올바르게 작동 |
| FlowMatching 캐시 빌드 | ✅ 정상 | QwenVLAUnified 사용 |
| Regression 캐시 빌드 | ⚠️ 작동하나 비권장 | DummyVLA 래퍼 사용 (취약) |
| TOTAL_TRAIN.sh VL 캐시 섹션 | ❌ 실행 불가 | 변수 미정의 + 주석 처리 |
| 경로 일관성 | ⚠️ 주의 필요 | CACHE_ROOT vs QWEN_CACHE_ROOT |

---

## 🚀 즉시 조치 사항

1. **TOTAL_TRAIN.sh 수정**: 변수 정의 및 VL 캐시 섹션 활성화
2. **경로 통일**: `QWEN_CACHE_ROOT="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features"` 명시적 정의
3. **테스트**: FlowMatching 캐시 빌드 먼저 실행하여 동작 확인

---

생성일: 2025-01-11
