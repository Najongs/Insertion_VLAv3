"""
VLA 캐시 관리자 (VLA Cache Manager)
- VLM 특징 캐싱을 위한 효율적인 관리 모듈
"""

import os
import shutil
from pathlib import Path
import fcntl
from typing import Optional, Dict, List

import torch


class VLACacheManager:
    """
    프롬프트 인식 캐싱을 위한 VLA 캐시 관리자.

    기능:
    - 프롬프트 해시를 사용하여 버전 관리된 캐시 경로 생성.
    - 안전한 동시 접근을 위한 Atomic Save.
    - 캐시 제한을 통한 디스크 공간 관리.
    """

    def __init__(
        self,
        cache_dir: str = "/dev/shm/vla_cache",
        cache_limit_gb: float = 50.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_limit_gb = cache_limit_gb

    def _raw_cache_path(self, dataset_name: str, vlm_idx: int, prompt_hash: str) -> Path:
        """
        주어진 정보로 캐시 파일의 경로를 생성합니다.
        """
        return (self.cache_dir / prompt_hash) / f"{dataset_name}_vlm{vlm_idx}.pt"

    def get_cache_path(
        self,
        dataset_name: str,
        vlm_idx: int,
        prompt_hash: str,
    ) -> Path:
        """
        프롬프트 인식 캐시 파일 경로를 생성하고, 해당 디렉토리가 존재하는지 확인합니다.
        """
        versioned_dir = self.cache_dir / prompt_hash
        versioned_dir.mkdir(parents=True, exist_ok=True) # 디렉토리 생성 보장
        return versioned_dir / f"{dataset_name}_vlm{vlm_idx}.pt"

    def cache_exists(
        self,
        dataset_name: str,
        vlm_idx: int,
        prompt_hash: str,
    ) -> bool:
        """
        주어진 프롬프트 해시에 대한 캐시 파일이 존재하는지 확인합니다.
        """
        cache_path = self._raw_cache_path(dataset_name, vlm_idx, prompt_hash)
        return cache_path.exists()

    def load_cache(
        self,
        dataset_name: str,
        vlm_idx: int,
        prompt_hash: str,
        device: str = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        특정 프롬프트 해시에 대한 캐시를 로드합니다.

        Returns:
            캐시된 VL 특징 텐서 또는 찾을 수 없는 경우 None.
        """
        cache_path = self._raw_cache_path(dataset_name, vlm_idx, prompt_hash)

        if not cache_path.exists():
            return None

        try:
            cached = torch.load(cache_path, map_location=device)
            return cached
        except Exception as e:
            print(f"⚠️ 캐시 로드 실패 {cache_path.name}: {e}")
            return None

    def save_cache(
        self,
        dataset_name: str,
        vlm_idx: int,
        prompt_hash: str,
        vl_features: torch.Tensor | tuple | list | dict,
    ):
        """
        특정 프롬프트 해시에 대한 캐시를 Atomic하게 저장합니다.
        단일 텐서, 튜플, 리스트, 딕셔너리를 모두 처리합니다.
        """
        cache_path = self.get_cache_path(dataset_name, vlm_idx, prompt_hash)

        # Process different input types
        def flatten_to_tensor(item, depth=0):
            """Helper to extract tensor from potentially nested tuple/list structure"""
            if depth > 5:  # Prevent infinite recursion
                raise TypeError(f"Too deeply nested structure (depth > 5)")

            if isinstance(item, torch.Tensor):
                return item.detach().to("cpu", dtype=torch.float16)
            elif isinstance(item, (tuple, list)):
                if len(item) == 0:
                    raise TypeError("Empty tuple/list cannot be converted to tensor")
                elif len(item) == 1:
                    return flatten_to_tensor(item[0], depth + 1)
                else:
                    # Multi-element: flatten each recursively
                    flattened = [flatten_to_tensor(x, depth + 1) for x in item]
                    return tuple(flattened) if isinstance(item, tuple) else flattened
            else:
                raise TypeError(f"Cannot convert {type(item)} to tensor (depth={depth})")

        # Convert vl_features to CPU tensors
        try:
            if isinstance(vl_features, dict):
                data_to_save = {k: flatten_to_tensor(v) for k, v in vl_features.items()}
            elif isinstance(vl_features, torch.Tensor):
                data_to_save = vl_features.detach().to("cpu", dtype=torch.float16)
            elif isinstance(vl_features, (tuple, list)):
                data_to_save = tuple(flatten_to_tensor(v) for v in vl_features)
            else:
                data_to_save = vl_features
        except (TypeError, AttributeError) as e:
            print(f"⚠️ Failed to process vl_features: {e}")
            print(f"   Type: {type(vl_features)}")
            if isinstance(vl_features, (tuple, list)):
                print(f"   Structure: {[type(v) for v in vl_features]}")
            return

        # 파일 락을 이용한 Atomic 저장
        self._atomic_save(data_to_save, cache_path)

        # 캐시 제한 적용
        self._enforce_cache_limit()

    @staticmethod
    def _atomic_save(tensor_cpu: torch.Tensor | tuple | list | dict, path: Path):
        """
        파일 락을 사용하여 Atomic하게 텐서(또는 텐서의 컬렉션)를 저장합니다.
        중간 파일(.pt.tmp)로 저장 후 원본 파일로 이동하여 파일 손상을 방지합니다.
        """
        tmp = path.with_suffix(".pt.tmp")
        lock_path = str(path) + ".lock"

        # 부모 디렉토리가 존재하는지 확인
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lockfile:
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX) # 배타적 락 획득

                # 다른 프로세스가 이미 저장했을 경우 건너뛰기
                if path.exists():
                    return

                torch.save(tensor_cpu, tmp) # 임시 파일에 저장
                os.replace(tmp, path) # 원본 파일로 이동 (atomic)

            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN) # 락 해제
                # 락 파일 정리
                try:
                    os.remove(lock_path)
                except OSError:
                    pass

    def _enforce_cache_limit(self):
        """
        모든 버전 관리된 하위 디렉토리에 걸쳐 캐시 크기 제한을 적용합니다.
        가장 오래된 파일부터 삭제합니다.
        """
        if self.cache_limit_gb <= 0:
            return

        all_files = list(self.cache_dir.rglob("*.pt")) # 모든 .pt 파일 찾기
        file_info = []
        total_bytes = 0
        for f in all_files:
            try:
                stat = f.stat()
            except FileNotFoundError:
                continue
            file_info.append((f, stat.st_size, stat.st_mtime))
            total_bytes += stat.st_size
        limit_bytes = self.cache_limit_gb * (1024 ** 3) # 설정된 캐시 한도 (바이트)

        if total_bytes <= limit_bytes: # 제한을 초과하지 않으면 종료
            return

        # 수정 시간 기준으로 파일 정렬 (가장 오래된 파일부터)
        file_info.sort(key=lambda item: item[2])

        # 제한을 초과하는 만큼 가장 오래된 파일부터 삭제
        for file, size, _ in file_info:
            if total_bytes <= limit_bytes:
                break
            try:
                file.unlink(missing_ok=True) # 파일 삭제
            except FileNotFoundError:
                continue
            total_bytes -= size

    def get_cache_stats(self) -> Dict:
        """
        전체 캐시에 대한 통계를 가져옵니다.
        """
        all_files = list(self.cache_dir.rglob("*.pt"))
        total_bytes = sum(f.stat().st_size for f in all_files)
        total_gb = total_bytes / (1024 ** 3) # GB 단위로 변환

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(all_files),
            "total_size_gb": total_gb,
            "limit_gb": self.cache_limit_gb,
            "usage_percent": (total_gb / self.cache_limit_gb * 100) if self.cache_limit_gb > 0 else 0,
        }

    def clear_cache(self, confirm: bool = False):
        """
        모든 하위 디렉토리를 포함하여 전체 캐시를 지웁니다.
        안전을 위해 `confirm=True`가 필요합니다.
        """
        if not confirm:
            print("⚠️ 캐시 지우려면 confirm=True가 필요합니다")
            return

        shutil.rmtree(self.cache_dir) # 디렉토리 및 모든 내용 삭제
        self.cache_dir.mkdir(parents=True, exist_ok=True) # 빈 디렉토리 재성성
        print(f"✅ {self.cache_dir}의 모든 캐시 파일과 하위 디렉토리를 지웠습니다")

    def list_cached_datasets(self) -> Dict:
        """
        `prompt_hash`별로 그룹화된 캐시된 데이터셋을 나열합니다.
        """
        all_files = list(self.cache_dir.rglob("*.pt"))
        
        versioned_datasets = {}
        for f in all_files:
            try:
                prompt_hash = f.parent.name
                name = f.stem
                if "_vlm" in name:
                    dataset_name, vlm_part = name.rsplit("_vlm", 1)
                    vlm_idx = int(vlm_part)

                    if prompt_hash not in versioned_datasets:
                        versioned_datasets[prompt_hash] = {}
                    if dataset_name not in versioned_datasets[prompt_hash]:
                        versioned_datasets[prompt_hash][dataset_name] = []
                    
                    versioned_datasets[prompt_hash][dataset_name].append(vlm_idx)
            except (ValueError, IndexError):
                continue

        # VLM 인덱스 정렬
        for prompt_hash, datasets in versioned_datasets.items():
            for dataset_name in datasets:
                datasets[dataset_name].sort()

        return versioned_datasets


# Global cache manager instance
_cache_manager = None


def get_cache_manager(
    cache_dir: str = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    cache_limit_gb: float = 50.0,
) -> VLACacheManager:
    """
    전역 캐시 관리자 인스턴스를 가져옵니다. 싱글톤 패턴과 유사하게 작동합니다.

    Args:
        cache_dir (str, optional): 캐시 저장 디렉토리. Defaults to "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features".
        cache_limit_gb (float, optional): 캐시의 최대 크기 (GB). Defaults to 50.0.

    Returns:
        VLACacheManager: 캐시 관리자 인스턴스.
    """
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = VLACacheManager(
            cache_dir=cache_dir,
            cache_limit_gb=cache_limit_gb,
        )
    # 필요한 경우 재구성 허용
    elif _cache_manager.cache_dir != Path(cache_dir) or _cache_manager.cache_limit_gb != cache_limit_gb:
         _cache_manager = VLACacheManager(
            cache_dir=cache_dir,
            cache_limit_gb=cache_limit_gb,
        )

    return _cache_manager
