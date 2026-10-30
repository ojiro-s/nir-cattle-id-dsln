"""疑似ラベル生成のテスト。特徴量はランダムなテンソルから作る。"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nir_cattle_id.pseudo_label import (
    DROPPED,
    assign_pseudo_labels,
    nearest_db_labels,
    save_pseudo_dataset,
    vote_in_clusters,
)

FEATURE_DIM = 64
NUM_IDS = 3


def make_features(num_per_id: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """個体ごとに固まったL2正規化済み特徴量と、個体IDを返す。"""
    generator = torch.Generator().manual_seed(0)
    centers = torch.randn(NUM_IDS, FEATURE_DIM, generator=generator)
    noise_generator = torch.Generator().manual_seed(seed)
    features = centers.repeat_interleave(num_per_id, dim=0)
    features += 0.05 * torch.randn(features.shape, generator=noise_generator)
    labels = np.repeat([f"id_{index}" for index in range(NUM_IDS)], num_per_id)
    return F.normalize(features, dim=1).numpy(), labels


def test_nearest_db_labels_uses_highest_similarity() -> None:
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    db = np.array([[0.0, 1.0], [1.0, 0.0]])
    labels = np.array(["a", "b"])

    assert nearest_db_labels(query, db, labels).tolist() == ["b", "a"]


def test_vote_returns_majority_and_purity() -> None:
    votes = np.array(["a", "a", "a", "b", "c", "c"])
    cluster_ids = np.array([0, 0, 0, 0, 1, 1])

    info = vote_in_clusters(votes, cluster_ids, n_clusters=3)

    assert info[0] == ("a", 0.75)
    assert info[1] == ("c", 1.0)
    assert info[2] == (DROPPED, 0.0)


def test_pseudo_labels_recover_separated_ids() -> None:
    db_features, db_labels = make_features(num_per_id=10, seed=1)
    query_features, query_labels = make_features(num_per_id=20, seed=2)

    pseudo_labels = assign_pseudo_labels(
        query_features, db_features, db_labels, n_clusters=6, purity_threshold=0.6,
    )

    assert pseudo_labels.tolist() == query_labels.tolist()


def test_low_purity_cluster_is_dropped() -> None:
    # 1つのクラスタに収まるNIR特徴量の最近傍を、DB側で a と b に半分ずつ割り振る
    base = np.array([1.0, 0.0, 0.0])
    query_features = np.stack([base + [0.0, offset, 0.0] for offset in (-0.01, -0.02, 0.01, 0.02)])
    db_features = np.array([[1.0, -0.02, 0.0], [1.0, 0.02, 0.0]])
    db_labels = np.array(["a", "b"])

    dropped = assign_pseudo_labels(
        query_features, db_features, db_labels, n_clusters=1, purity_threshold=0.6,
    )
    kept = assign_pseudo_labels(
        query_features, db_features, db_labels, n_clusters=1, purity_threshold=0.5,
    )

    assert dropped.tolist() == [DROPPED] * 4
    assert DROPPED not in kept.tolist()


def test_random_features_get_known_labels_or_dropped() -> None:
    query_features = F.normalize(torch.randn(50, FEATURE_DIM), dim=1).numpy()
    db_features, db_labels = make_features(num_per_id=5, seed=3)

    pseudo_labels = assign_pseudo_labels(
        query_features, db_features, db_labels, n_clusters=10, purity_threshold=0.6,
    )

    assert pseudo_labels.shape == (50,)
    assert set(pseudo_labels) <= set(db_labels) | {DROPPED}


def test_save_skips_dropped_images(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    paths = [source / name for name in ("kept.jpg", "dropped.jpg")]
    for path in paths:
        path.write_bytes(b"image")
    output = tmp_path / "output"

    copied = save_pseudo_dataset(["id_0", DROPPED], [str(path) for path in paths], output)

    assert copied == 1
    assert (output / "id_0" / "kept.jpg").exists()
    assert not (output / DROPPED).exists()
