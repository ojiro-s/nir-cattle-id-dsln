"""画像前処理とPKサンプリングのDataLoader。"""

import logging
from dataclasses import dataclass

import torchvision.transforms as T
from pytorch_metric_learning import samplers
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class AugmentConfig:
    """学習時のデータ拡張のパラメータ。"""

    degrees: float
    scale: tuple[float, float]
    shear: float
    brightness: float
    contrast: float
    blur_kernel_size: int
    blur_sigma: tuple[float, float]


def get_train_transforms(size: int, augment: AugmentConfig, p_gsb: float) -> T.Compose:
    """学習用の前処理を返す。p_gsb はグレースケール+ぼかしを適用する確率。"""
    grayscale_blur = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.GaussianBlur(
            kernel_size=(augment.blur_kernel_size, augment.blur_kernel_size),
            sigma=augment.blur_sigma,
        ),
    ])
    return T.Compose([
        T.Resize((size, size)),
        T.RandomAffine(degrees=augment.degrees, scale=augment.scale, shear=augment.shear),
        T.ColorJitter(brightness=augment.brightness, contrast=augment.contrast),
        T.RandomApply([grayscale_blur], p=p_gsb),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_test_transforms(size: int) -> T.Compose:
    """推論用の前処理を返す。"""
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def align_labels(dataset: ImageFolder, reference: ImageFolder) -> None:
    """NIRデータセットのラベル番号をRGBデータセットの個体IDに揃える。

    疑似ラベル付きNIRには全個体が含まれるとは限らず、ImageFolderが振る番号が
    RGBとずれるため、ArcFaceの共有クラス重みに合わせて振り直す。
    """
    samples = []
    for path, label in dataset.samples:
        class_name = dataset.classes[label]
        if class_name in reference.class_to_idx:
            samples.append((path, reference.class_to_idx[class_name]))
        else:
            logger.warning("Class %s not found in the reference dataset. Skipping.", class_name)
    dataset.samples = samples
    dataset.targets = [label for _, label in samples]
    dataset.classes = reference.classes
    dataset.class_to_idx = reference.class_to_idx


def create_pk_loader(dataset: ImageFolder, p: int, k: int, num_workers: int) -> DataLoader:
    """P個のクラスからK枚ずつ取り出すバッチのDataLoaderを返す。"""
    batch_size = p * k
    sampler = samplers.MPerClassSampler(
        labels=dataset.targets,
        m=k,
        batch_size=batch_size,
        length_before_new_iter=len(dataset),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
