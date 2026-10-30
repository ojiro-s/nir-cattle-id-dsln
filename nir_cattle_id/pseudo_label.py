"""NIR画像の疑似ラベル生成: Over-clustering、最近傍投票、純度による棄却。"""

import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

DROPPED = "Dropped"


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """L2正規化した特徴量と、DataLoaderが返すラベル番号を取り出す。"""
    model.eval()
    features, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            feature = F.normalize(model(images.to(device)), p=2, dim=1)
            features.append(feature.cpu())
            targets.append(labels)
    return torch.cat(features).numpy(), torch.cat(targets).numpy()


def nearest_db_labels(
    query_features: np.ndarray,
    db_features: np.ndarray,
    db_labels: np.ndarray,
) -> np.ndarray:
    """各クエリに、コサイン類似度が最大のDB画像のラベル(k=1)を割り当てる。"""
    similarity = np.dot(query_features, db_features.T)
    return db_labels[np.argmax(similarity, axis=1)]


def vote_in_clusters(
    votes: np.ndarray,
    cluster_ids: np.ndarray,
    n_clusters: int,
) -> dict[int, tuple[str, float]]:
    """クラスタごとに多数決を取り、(多数派ラベル, 純度) を返す。"""
    cluster_info = {}
    for cluster_id in range(n_clusters):
        members = votes[cluster_ids == cluster_id]
        if len(members) == 0:
            cluster_info[cluster_id] = (DROPPED, 0.0)
            continue
        majority, count = Counter(members).most_common(1)[0]
        cluster_info[cluster_id] = (majority, count / len(members))
    return cluster_info


def assign_pseudo_labels(
    query_features: np.ndarray,
    db_features: np.ndarray,
    db_labels: np.ndarray,
    n_clusters: int,
    purity_threshold: float,
    n_init: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """NIR特徴量に疑似ラベルを付ける。純度が閾値未満のクラスタは DROPPED にする。

    Args:
        query_features: L2正規化済みのNIR特徴量 (N, D)。
        db_features: L2正規化済みのRGBデータベース特徴量 (M, D)。
        db_labels: データベース画像の個体ID (M,)。
        n_clusters: K-Meansのクラスタ数。個体数より多くとる(Over-clustering)。
        purity_threshold: クラスタを採用する多数派の割合の下限。
        n_init: K-Meansの初期値を変えて試す回数。
        random_state: K-Meansの乱数シード。

    Returns:
        各NIR画像の疑似ラベル (N,)。
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
    )
    cluster_ids = kmeans.fit_predict(query_features)
    votes = nearest_db_labels(query_features, db_features, db_labels)
    cluster_info = vote_in_clusters(votes, cluster_ids, n_clusters)

    pseudo_labels = []
    for cluster_id in cluster_ids:
        label, purity = cluster_info[cluster_id]
        pseudo_labels.append(label if purity >= purity_threshold else DROPPED)
    return np.array(pseudo_labels)


def save_pseudo_dataset(
    pseudo_labels: Sequence[str],
    image_paths: Sequence[str],
    output_dir: Union[str, Path],
) -> int:
    """採用された画像を <output_dir>/<個体ID>/ にコピーし、枚数を返す。"""
    copy_count = 0
    for src_path, label in zip(image_paths, pseudo_labels):
        if label == DROPPED:
            continue
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        dst_path = os.path.join(class_dir, os.path.basename(src_path))
        try:
            shutil.copy2(src_path, dst_path)
            copy_count += 1
        except OSError as error:
            logger.warning("Copy failed: %s -> %s (%s)", src_path, dst_path, error)
    return copy_count
