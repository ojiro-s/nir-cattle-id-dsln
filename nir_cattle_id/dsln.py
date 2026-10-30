"""Domain-Specific LayerNorm (DSLN) の実装とモデルへの組み込み。"""

from typing import Literal

import torch
import torch.nn as nn

Domain = Literal["rgb", "nir"]


class DSLN(nn.Module):
    """RGB用とNIR用の2つのLayerNormを持ち、現在のドメインに応じて切り替える層。"""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.ln_rgb = nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine)
        self.ln_nir = nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine)
        self.domain: Domain = "rgb"

    def set_domain(self, domain: Domain) -> None:
        """以降の順伝播で使うLayerNormを切り替える。"""
        assert domain in ("rgb", "nir")
        self.domain = domain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """現在のドメインのLayerNormを適用する。"""
        if self.domain == "rgb":
            return self.ln_rgb(x)
        return self.ln_nir(x)


def set_model_domain(model: nn.Module, domain: Domain) -> None:
    """モデル内の全DSLN層のドメインを切り替える。"""
    count = 0
    for module in model.modules():
        if isinstance(module, DSLN):
            module.set_domain(domain)
            count += 1
    if count == 0:
        raise RuntimeError("No DSLN layers found in the model.")


def convert_to_dsln(model: nn.Module) -> nn.Module:
    """モデル内の全 nn.LayerNorm を DSLN に置換し、元の重みを両ドメインにコピーする。"""
    target_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
    ]
    if not target_modules:
        raise RuntimeError("No LayerNorm modules found to convert.")

    for name, module in target_modules:
        # "layers.0.blocks.0.norm1" のようなドット区切りの名前から親モジュールをたどる
        *parent_path, target_attr = name.split(".")
        parent = model
        for atom in parent_path:
            parent = getattr(parent, atom)

        num_features = (
            module.normalized_shape
            if isinstance(module.normalized_shape, int)
            else module.normalized_shape[0]
        )
        dsln = DSLN(num_features, eps=module.eps, affine=module.elementwise_affine)

        if module.elementwise_affine:
            with torch.no_grad():
                if module.weight is not None:
                    dsln.ln_rgb.weight.copy_(module.weight)
                    dsln.ln_nir.weight.copy_(module.weight)
                if module.bias is not None:
                    dsln.ln_rgb.bias.copy_(module.bias)
                    dsln.ln_nir.bias.copy_(module.bias)

        setattr(parent, target_attr, dsln)

    return model
