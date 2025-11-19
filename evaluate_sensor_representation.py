"""
센서 인코더 표현력 진단 스크립트
----------------------------------

기능:
1. UnifiedGatedSensorEncoder 체크포인트를 로드하여 센서 데이터 임베딩을 수집
2. PCA / t-SNE 투영 결과를 시각화하여 episode 진행도 또는 접촉 레이블과 비교
3. 간단한 K-Means / 로지스틱 회귀 기반 품질 지표 기록
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from models.Encoder_model import UnifiedGatedSensorEncoder
from vla_datasets.unified_dataset import create_unified_dataloader


class SensorEmbeddingEvalDataset(Dataset):
    """
    Unified 데이터셋을 순회하면서 episode 진행도를 기준으로 contact pseudo-label을 생성.
    Sensor 데이터만 반환하여 빠른 표현력 분석에 사용한다.
    """

    def __init__(self, unified_dataset, contact_threshold: float = 0.8):
        self.unified_dataset = unified_dataset
        self.contact_threshold = contact_threshold
        self.samples: List[Dict] = []
        self._prepare_samples()

    def _prepare_samples(self):
        global_idx_offset = 0
        for ds in self.unified_dataset.datasets:
            episode_len = len(ds)
            if episode_len == 0:
                continue
            for local_idx in range(episode_len):
                global_idx = global_idx_offset + local_idx
                relative_pos = (
                    local_idx / (episode_len - 1) if episode_len > 1 else 1.0
                )
                is_contact = relative_pos >= self.contact_threshold
                self.samples.append(
                    {
                        "global_idx": global_idx,
                        "relative_pos": relative_pos,
                        "contact_label": 1.0 if is_contact else 0.0,
                    }
                )
            global_idx_offset += episode_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        data = self.unified_dataset[sample_info["global_idx"]]
        sensor_tensor = data["sensor_data"]
        if not torch.is_tensor(sensor_tensor):
            sensor_tensor = torch.tensor(sensor_tensor, dtype=torch.float32)
        return {
            "sensor_data": sensor_tensor,
            "contact_label": torch.tensor(sample_info["contact_label"], dtype=torch.float32),
            "relative_pos": torch.tensor(sample_info["relative_pos"], dtype=torch.float32),
            "episode_id": data.get("episode_id", "unknown"),
            "vlm_idx": data.get("vlm_idx", 0),
            "confusion_matrix": [[float("nan"), float("nan")], [float("nan"), float("nan")]],
        }


def sensor_eval_collate(batch):
    sensor_tensors = [item["sensor_data"] for item in batch]
    lengths = [t.shape[0] for t in sensor_tensors]
    max_len = max(lengths)
    padded = []
    for tensor in sensor_tensors:
        if tensor.shape[0] < max_len:
            pad = torch.zeros(max_len - tensor.shape[0], tensor.shape[1], dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad], dim=0)
        padded.append(tensor)
    sensor_batch = torch.stack(padded, dim=0)

    contact = torch.stack([item["contact_label"] for item in batch])
    rel = torch.stack([item["relative_pos"] for item in batch])
    meta = {
        "episode_id": [item["episode_id"] for item in batch],
        "vlm_idx": torch.tensor([item["vlm_idx"] for item in batch], dtype=torch.float32),
    }
    return sensor_batch, contact, rel, meta


def load_sensor_encoder(
    checkpoint_path: Path,
    device: torch.device,
    encoder_kwargs: Dict,
) -> UnifiedGatedSensorEncoder:
    model = UnifiedGatedSensorEncoder(**encoder_kwargs)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "encoder_state_dict" in ckpt:
        state_dict = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = {
            k.replace("sensor_encoder.", "", 1): v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("sensor_encoder.")
        }
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def collect_embeddings(
    encoder: UnifiedGatedSensorEncoder,
    dataloader: DataLoader,
    num_samples: int,
    device: torch.device,
):
    embeddings = []
    labels = []
    rel_pos_list = []
    episodes = []
    collected = 0
    with torch.no_grad():
        for sensor_batch, contact, rel, meta in dataloader:
            sensor_batch = sensor_batch.to(device=device, dtype=torch.float32)
            global_feat, _ = encoder(sensor_batch)
            embeddings.append(global_feat.cpu().float())
            labels.append(contact.cpu().float())
            rel_pos_list.append(rel.cpu().float())
            episodes.extend(meta["episode_id"])
            collected += sensor_batch.size(0)
            if collected >= num_samples:
                break
    emb = torch.cat(embeddings, dim=0)[:num_samples]
    lab = torch.cat(labels, dim=0)[:num_samples]
    rel = torch.cat(rel_pos_list, dim=0)[:num_samples]
    episodes = episodes[:num_samples]
    return emb.numpy(), lab.numpy(), rel.numpy(), episodes


def run_linear_probe(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    y = (labels > 0.5).astype(int)
    if len(np.unique(y)) < 2:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "tn": float("nan"),
            "fp": float("nan"),
            "fn": float("nan"),
            "tp": float("nan"),
        }
    split = int(len(embeddings) * 0.8)
    X_train, X_test = embeddings[:split], embeddings[split:]
    y_train, y_test = y[:split], y[split:]
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = roc_auc_score(y_test, prob)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "confusion_matrix": cm.tolist(),
    }


def visualize_embeddings(embeddings, rel_pos, labels, output_dir: Path, use_tsne: bool):
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=rel_pos, cmap="viridis", s=8)
    plt.colorbar(label="Episode Progress")
    plt.title("Sensor Embeddings PCA (Colored by Progress)")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_progress.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        pca_coords[:, 0],
        pca_coords[:, 1],
        c=(labels > 0.5).astype(float),
        cmap="coolwarm",
        s=8,
    )
    plt.colorbar(label="Contact Label")
    plt.title("Sensor Embeddings PCA (Contact)")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_contact.png", dpi=200)
    plt.close()

    if use_tsne:
        tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto")
        tsne_coords = tsne.fit_transform(embeddings)
        plt.figure(figsize=(6, 5))
        plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=rel_pos, cmap="viridis", s=8)
        plt.colorbar(label="Episode Progress")
        plt.title("Sensor Embeddings t-SNE (Progress)")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_progress.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.scatter(
            tsne_coords[:, 0],
            tsne_coords[:, 1],
            c=(labels > 0.5).astype(float),
            cmap="coolwarm",
            s=8,
        )
        plt.colorbar(label="Contact Label")
        plt.title("Sensor Embeddings t-SNE (Contact)")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_contact.png", dpi=200)
        plt.close()


def save_confusion_matrix(cm: np.ndarray, output_dir: Path):
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Linear Probe Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:.0f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(output_dir / "linear_probe_confusion_matrix.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="센서 인코더 표현력 진단 스크립트")
    parser.add_argument("--new_dataset_paths", type=str, nargs="+", required=True)
    parser.add_argument("--sensor_checkpoint", type=str, required=True)
    parser.add_argument("--cache_root", type=str, default="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/sensor_representation")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--contact_threshold", type=float, default=0.8)
    parser.add_argument("--tsne", action="store_true", help="t-SNE 시각화 포함 여부")
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--sensor_output_dim", type=int, default=3072)
    parser.add_argument("--sensor_dist_channels", type=int, default=1025)
    parser.add_argument("--sensor_force_channels", type=int, default=1)
    parser.add_argument("--sensor_temporal_length", type=int, default=65)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Unified 데이터셋 로딩 중...")
    unified_dataset = create_unified_dataloader(
        new_dataset_paths=args.new_dataset_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_dataset=True,
        use_cache=False,
        skip_dataset_stats=True,
        disable_robot_state=True,
        disable_sensor=False,
    )
    eval_dataset = SensorEmbeddingEvalDataset(unified_dataset, contact_threshold=args.contact_threshold)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=sensor_eval_collate,
        pin_memory=True,
    )

    print("🤖 센서 인코더 로딩 중...")
    encoder_kwargs = dict(
        output_dim=args.sensor_output_dim,
        dist_channels=args.sensor_dist_channels,
        force_channels=args.sensor_force_channels,
        temporal_length=args.sensor_temporal_length,
    )
    sensor_encoder = load_sensor_encoder(Path(args.sensor_checkpoint), device, encoder_kwargs)

    print("📊 임베딩 수집 중...")
    embeddings, labels, rel_pos, episodes = collect_embeddings(
        sensor_encoder, dataloader, args.num_samples, device
    )
    np.savez(output_dir / "sensor_embeddings.npz", embeddings=embeddings, labels=labels, rel_pos=rel_pos)

    print("📉 차원 축소 시각화 생성...")
    visualize_embeddings(embeddings, rel_pos, labels, output_dir, use_tsne=args.tsne)

    print("🧪 로지스틱 회귀 기반 선형 분리 성능 측정...")
    probe_metrics = run_linear_probe(embeddings, labels)
    if probe_metrics["confusion_matrix"]:
        cm_array = np.array(probe_metrics["confusion_matrix"])
        save_confusion_matrix(cm_array, output_dir)

    print("🌀 K-Means 클러스터링...")
    kmeans = KMeans(n_clusters=args.num_clusters, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    ami = adjusted_mutual_info_score((labels > 0.5).astype(int), cluster_ids)

    metrics = {
        "linear_probe": probe_metrics,
        "ami_contact_vs_cluster": ami,
        "num_samples": int(embeddings.shape[0]),
        "contact_threshold": args.contact_threshold,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ 평가 완료. 결과 경로:", output_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
