"""Swin Transformer の構築とDSLNの組み込み。"""

from pathlib import Path
from typing import Union

import timm
import torch
import torch.nn as nn

from .dsln import DSLN, convert_to_dsln, set_model_domain


def load_state_dict(checkpoint_path: Union[str, Path]) -> dict[str, torch.Tensor]:
    """チェックポイントを読み込み、DataParallelの "module." 接頭辞を除いて返す。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def load_backbone(
    model_name: str,
    checkpoint_path: Union[str, Path],
    device: str,
) -> nn.Module:
    """疑似ラベル生成に使う、標準LayerNormの学習済みモデルを推論モードで返す。"""
    model = timm.create_model(model_name, num_classes=0, pretrained=False)
    model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
    return model.to(device).eval()


def freeze_except_finetune_layers(model: nn.Module) -> None:
    """全層を凍結し、Swinのステージ3・4と全DSLN層だけを学習対象にする。"""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layers[2:4].parameters():
        param.requires_grad = True
    for module in model.modules():
        if isinstance(module, DSLN):
            for param in module.parameters():
                param.requires_grad = True


def build_dsln_swin(
    model_name: str,
    base_checkpoint: Union[str, Path, None] = None,
    pretrained: bool = True,
) -> nn.Module:
    """RGB学習済みの重みを読み込んだSwinをDSLN化し、凍結設定まで済ませて返す。

    標準LayerNormの重みを読むため、ロードはDSLNへの置換より先に行う。
    """
    model = timm.create_model(model_name, num_classes=0, pretrained=pretrained)
    if base_checkpoint is not None:
        model.load_state_dict(load_state_dict(base_checkpoint), strict=True)
    model = convert_to_dsln(model)
    freeze_except_finetune_layers(model)
    return model


def embedding_dim(model: nn.Module, input_size: int, device: str) -> int:
    """ダミー入力を流して埋め込みの次元数を返す。"""
    target = model.module if isinstance(model, nn.DataParallel) else model
    set_model_domain(target, "rgb")
    with torch.no_grad():
        dummy = torch.randn(1, 3, input_size, input_size).to(device)
        return model(dummy).shape[1]
