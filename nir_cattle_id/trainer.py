"""RGBバッチとNIRバッチを交互に流す学習ループ。"""

import csv
import gc
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dsln import Domain, set_model_domain

logger = logging.getLogger(__name__)


def top1_accuracy(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    db_features: np.ndarray,
    db_labels: np.ndarray,
) -> float:
    """データベースの最近傍(k=1)のラベルが正解と一致する割合を返す。"""
    neighbors = KNeighborsClassifier(n_neighbors=1)
    neighbors.fit(db_features, db_labels)
    _, indices = neighbors.kneighbors(query_features, n_neighbors=1)
    return float(np.mean(db_labels[indices[:, 0]] == query_labels))


class DualDomainTrainer:
    """ドメインごとにDSLNを切り替え、ArcFace損失を共有して学習する。"""

    def __init__(
        self,
        model: nn.Module,
        objective: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        loader_rgb: DataLoader,
        loader_nir: DataLoader,
        loader_db: DataLoader,
        loader_val_rgb: DataLoader,
        loader_val_nir: DataLoader,
        device: str,
        total_epochs: int,
        accumulation_steps: int,
        val_interval: int,
        save_dir: str,
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader_rgb = loader_rgb
        self.loader_nir = loader_nir
        self.loader_db = loader_db
        self.loader_val_rgb = loader_val_rgb
        self.loader_val_nir = loader_val_nir
        self.device = device
        self.total_epochs = total_epochs
        self.accumulation_steps = accumulation_steps
        self.val_interval = val_interval
        self.save_dir = save_dir
        self.amp_enabled = device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        self.epoch = 0
        self.best_sum = 0.0
        self.log_file = os.path.join(save_dir, "train_log.csv")

        if not os.path.exists(self.log_file):
            self._append_log(["epoch", "train_loss", "val_rgb_acc", "val_nir_acc"])

    def _unwrap_model(self) -> nn.Module:
        """DataParallelを外した本体を返す。"""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _append_log(self, row: list) -> None:
        """学習ログのCSVに1行追記する。"""
        with open(self.log_file, "a", newline="") as file:
            csv.writer(file).writerow(row)

    def _forward_backward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        domain: Domain,
    ) -> torch.Tensor:
        """指定ドメインのDSLNで順伝播し、勾配を累積して損失を返す。"""
        images, labels = images.to(self.device), labels.to(self.device)
        set_model_domain(self._unwrap_model(), domain)
        with torch.amp.autocast(device_type=self.device, enabled=self.amp_enabled):
            embeddings = self.model(images)
            loss = self.objective(embeddings, labels) / self.accumulation_steps
        self.scaler.scale(loss).backward()
        return loss

    def train_step(
        self,
        step: int,
        batch_rgb: tuple[torch.Tensor, torch.Tensor],
        batch_nir: tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """RGBとNIRを1バッチずつ流し、更新時期なら重みを更新して合計損失を返す。"""
        loss_rgb = self._forward_backward(*batch_rgb, "rgb")
        loss_nir = self._forward_backward(*batch_nir, "nir")

        if (step + 1) % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return (loss_rgb.item() + loss_nir.item()) * self.accumulation_steps

    def train(self) -> None:
        """全エポックを学習する。1エポックの長さは短い方のDataLoaderに合わせる。"""
        for _ in range(self.epoch, self.total_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            total_loss = 0.0
            num_batches = min(len(self.loader_rgb), len(self.loader_nir))

            progress = tqdm(
                zip(self.loader_rgb, self.loader_nir),
                total=num_batches,
                desc=f"Epoch {self.epoch + 1}/{self.total_epochs}",
            )
            for step, (batch_rgb, batch_nir) in enumerate(progress):
                loss = self.train_step(step, batch_rgb, batch_nir)
                total_loss += loss
                progress.set_postfix({"loss": f"{loss:.4f}"})

            if self.scheduler:
                self.scheduler.step()

            self.epoch += 1
            avg_loss = total_loss / num_batches

            if self.epoch % self.val_interval == 0 or self.epoch == self.total_epochs:
                val_rgb_acc, val_nir_acc = self.validate()
                logger.info(
                    "Epoch %d: loss %.4f, RGB acc %.4f, NIR acc %.4f",
                    self.epoch, avg_loss, val_rgb_acc, val_nir_acc,
                )
                self._append_log([self.epoch, avg_loss, val_rgb_acc, val_nir_acc])
                self.save_checkpoint(val_rgb_acc, val_nir_acc)

    def _extract_features(
        self,
        loader: DataLoader,
        domain: Domain,
    ) -> tuple[np.ndarray, np.ndarray]:
        """指定ドメインのDSLNで特徴量を抽出し、個体IDの文字列と組で返す。"""
        set_model_domain(self._unwrap_model(), domain)
        features, labels = [], []
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=f"Extracting {domain}", leave=False):
                feature = F.normalize(self.model(images.to(self.device)), p=2, dim=1)
                features.append(feature.cpu())
                labels.append(targets)

        # データセットごとにラベル番号が異なるため、個体IDの文字列で比較する
        indices = torch.cat(labels).numpy()
        class_names = np.array([loader.dataset.classes[index] for index in indices])
        return torch.cat(features).numpy(), class_names

    def validate(self) -> tuple[float, float]:
        """RGBデータベースに対する、RGB・NIRクエリのTop-1精度を返す。"""
        self.model.eval()
        db_features, db_labels = self._extract_features(self.loader_db, "rgb")
        rgb_features, rgb_labels = self._extract_features(self.loader_val_rgb, "rgb")
        nir_features, nir_labels = self._extract_features(self.loader_val_nir, "nir")

        acc_rgb = top1_accuracy(rgb_features, rgb_labels, db_features, db_labels)
        acc_nir = top1_accuracy(nir_features, nir_labels, db_features, db_labels)

        gc.collect()
        torch.cuda.empty_cache()
        return acc_rgb, acc_nir

    def save_checkpoint(self, acc_rgb: float, acc_nir: float) -> None:
        """RGBとNIRの精度の和が最良のモデルと、検証時点のモデルを保存する。"""
        current_sum = acc_rgb + acc_nir
        if current_sum > self.best_sum:
            self.best_sum = current_sum
            self.save(os.path.join(self.save_dir, "best_sum.pth"))
        self.save(os.path.join(self.save_dir, f"epoch_{self.epoch}.pth"))

    def save(self, path: str) -> None:
        """モデル・損失・オプティマイザの状態を保存する。"""
        torch.save({
            "model": self._unwrap_model().state_dict(),
            "objective": self.objective.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }, path)
