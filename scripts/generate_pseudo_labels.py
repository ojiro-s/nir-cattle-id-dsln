"""NIR画像に疑似ラベルを付け、ImageFolder形式のデータセットとして保存する。

使い方: python -m scripts.generate_pseudo_labels --config configs/pseudo_label.yaml
"""

import argparse
import logging

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from nir_cattle_id.data import get_test_transforms
from nir_cattle_id.model import load_backbone
from nir_cattle_id.pseudo_label import (
    assign_pseudo_labels,
    extract_features,
    save_pseudo_dataset,
)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解釈する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML設定ファイルのパス")
    return parser.parse_args()


def main() -> None:
    """特徴抽出、疑似ラベル付け、データセット保存を順に行う。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open(parse_args().config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(config["model"]["name"], config["model"]["checkpoint"], device)
    transform = get_test_transforms(config["model"]["input_size"])

    # NIR側のフォルダ分けは読み込みの都合で必要なだけで、クラス名は使わない
    db_dataset = ImageFolder(config["data"]["rgb_db_dir"], transform=transform)
    nir_dataset = ImageFolder(config["data"]["nir_dir"], transform=transform)
    db_loader, nir_loader = (
        DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )
        for dataset in (db_dataset, nir_dataset)
    )

    db_features, db_targets = extract_features(model, db_loader, device)
    nir_features, _ = extract_features(model, nir_loader, device)
    db_labels = np.array([db_dataset.classes[index] for index in db_targets])

    pseudo_labels = assign_pseudo_labels(
        nir_features,
        db_features,
        db_labels,
        n_clusters=config["clustering"]["n_clusters"],
        purity_threshold=config["clustering"]["purity_threshold"],
        n_init=config["clustering"]["n_init"],
        random_state=config["clustering"]["random_state"],
    )
    image_paths = [path for path, _ in nir_dataset.samples]
    copied = save_pseudo_dataset(pseudo_labels, image_paths, config["data"]["output_dir"])
    logging.info("Saved %d / %d images.", copied, len(image_paths))


if __name__ == "__main__":
    main()
