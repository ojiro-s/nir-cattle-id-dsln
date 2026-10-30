"""RGBデータと疑似ラベル付きNIRデータで、DSLN付きSwinを追加学習する。

使い方: python -m scripts.train --config configs/train_dsln.yaml --seed 42
"""

import argparse
import logging
import os
import random
from itertools import chain
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from nir_cattle_id.data import (
    AugmentConfig,
    align_labels,
    create_pk_loader,
    get_test_transforms,
    get_train_transforms,
)
from nir_cattle_id.model import build_dsln_swin, embedding_dim
from nir_cattle_id.trainer import DualDomainTrainer


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解釈する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML設定ファイルのパス")
    parser.add_argument("--seed", type=int, default=None, help="設定ファイルのseedを上書きする")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Python・NumPy・PyTorchの乱数シードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_eval_loader(
    root: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """推論用の前処理をかけた、順序固定のDataLoaderを返す。"""
    dataset = ImageFolder(root, transform=get_test_transforms(input_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main(config: dict, seed: Optional[int]) -> None:
    """データ、モデル、損失、オプティマイザを組み立てて学習を実行する。"""
    set_seed(config["seed"] if seed is None else seed)
    save_dir = config["output_dir"]
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = config["data"]
    model_config = config["model"]
    input_size = model_config["input_size"]
    p, k = config["sampler"]["p"], config["sampler"]["k"]
    batch_size = p * k
    num_workers = config["num_workers"]

    augment_config = dict(config["augmentation"])
    p_gsb_rgb = augment_config.pop("p_gsb_rgb")
    p_gsb_nir = augment_config.pop("p_gsb_nir")
    augment_config["scale"] = tuple(augment_config["scale"])
    augment_config["blur_sigma"] = tuple(augment_config["blur_sigma"])
    augment = AugmentConfig(**augment_config)

    dataset_rgb = ImageFolder(
        data["train_rgb_dir"],
        transform=get_train_transforms(input_size, augment, p_gsb_rgb),
    )
    dataset_nir = ImageFolder(
        data["train_nir_dir"],
        transform=get_train_transforms(input_size, augment, p_gsb_nir),
    )
    align_labels(dataset_nir, dataset_rgb)

    loader_rgb = create_pk_loader(dataset_rgb, p, k, num_workers)
    loader_nir = create_pk_loader(dataset_nir, p, k, num_workers)
    loader_db = create_eval_loader(data["train_rgb_dir"], input_size, batch_size, num_workers)
    loader_val_rgb = create_eval_loader(data["val_rgb_dir"], input_size, batch_size, num_workers)
    loader_val_nir = create_eval_loader(data["val_nir_dir"], input_size, batch_size, num_workers)

    model = build_dsln_swin(
        model_config["name"],
        base_checkpoint=model_config["base_checkpoint"],
        pretrained=model_config["pretrained"],
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=model_config["device_ids"])

    objective = losses.ArcFaceLoss(
        num_classes=len(dataset_rgb.classes),
        embedding_size=embedding_dim(model, input_size, device),
        margin=config["loss"]["margin"],
        scale=config["loss"]["scale"],
    )
    params = chain(
        filter(lambda param: param.requires_grad, model.parameters()),
        objective.parameters(),
    )
    optimizer = torch.optim.AdamW(
        params,
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    total_epochs = config["trainer"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=config["scheduler"]["eta_min"],
    )

    trainer = DualDomainTrainer(
        model,
        objective,
        optimizer,
        scheduler,
        loader_rgb,
        loader_nir,
        loader_db,
        loader_val_rgb,
        loader_val_nir,
        device=device,
        total_epochs=total_epochs,
        accumulation_steps=config["trainer"]["accumulation_steps"],
        val_interval=config["trainer"]["val_interval"],
        save_dir=save_dir,
    )
    trainer.train()
    trainer.save(str(Path(save_dir) / "final.pth"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        main(yaml.safe_load(file), args.seed)
