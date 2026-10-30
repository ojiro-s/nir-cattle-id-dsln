"""DSLN付きSwinの順伝播と学習1ステップのテスト。実データも通信も使わない。"""

from pathlib import Path

import pytest
import timm
import torch
import torch.nn as nn
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader

from nir_cattle_id.dsln import DSLN, convert_to_dsln, set_model_domain
from nir_cattle_id.model import build_dsln_swin
from nir_cattle_id.trainer import DualDomainTrainer

MODEL_NAME = "swin_tiny_patch4_window7_224"
INPUT_SIZE = 224


@pytest.fixture
def swin() -> nn.Module:
    """DSLN化する前の、ランダム初期化されたSwinを返す。"""
    torch.manual_seed(0)
    return timm.create_model(MODEL_NAME, num_classes=0, pretrained=False).eval()


def count_layer_norms(model: nn.Module) -> int:
    """DSLNの内部以外にある nn.LayerNorm の数を返す。"""
    inside_dsln = {
        id(child)
        for module in model.modules() if isinstance(module, DSLN)
        for child in module.children()
    }
    return sum(
        isinstance(module, nn.LayerNorm) and id(module) not in inside_dsln
        for module in model.modules()
    )


def test_convert_replaces_all_layer_norms(swin: nn.Module) -> None:
    num_layer_norms = count_layer_norms(swin)
    assert num_layer_norms > 0

    model = convert_to_dsln(swin)

    assert count_layer_norms(model) == 0
    assert sum(isinstance(module, DSLN) for module in model.modules()) == num_layer_norms


def test_forward_returns_embedding_for_both_domains(swin: nn.Module) -> None:
    expected_dim = swin.num_features
    model = convert_to_dsln(swin)
    images = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    for domain in ("rgb", "nir"):
        set_model_domain(model, domain)
        with torch.no_grad():
            assert model(images).shape == (2, expected_dim)


def test_converted_model_keeps_output_until_trained(swin: nn.Module) -> None:
    images = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        expected = swin(images)

    model = convert_to_dsln(swin)

    for domain in ("rgb", "nir"):
        set_model_domain(model, domain)
        with torch.no_grad():
            assert torch.allclose(model(images), expected)


def test_domain_selects_its_own_layer_norm(swin: nn.Module) -> None:
    model = convert_to_dsln(swin)
    for module in model.modules():
        if isinstance(module, DSLN):
            nn.init.constant_(module.ln_nir.weight, 2.0)
    images = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        set_model_domain(model, "rgb")
        output_rgb = model(images)
        set_model_domain(model, "nir")
        output_nir = model(images)

    assert not torch.allclose(output_rgb, output_nir)


def test_set_model_domain_requires_dsln() -> None:
    with pytest.raises(RuntimeError):
        set_model_domain(nn.Linear(2, 2), "nir")


def test_build_loads_checkpoint_saved_with_data_parallel(swin: nn.Module, tmp_path: Path) -> None:
    state_dict = {f"module.{key}": value for key, value in swin.state_dict().items()}
    checkpoint = tmp_path / "base.pth"
    torch.save({"model": state_dict}, checkpoint)
    images = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        expected = swin(images)

    model = build_dsln_swin(MODEL_NAME, base_checkpoint=checkpoint, pretrained=False).eval()

    set_model_domain(model, "nir")
    with torch.no_grad():
        assert torch.allclose(model(images), expected)


def test_build_trains_only_late_stages_and_dsln() -> None:
    model = build_dsln_swin(MODEL_NAME, pretrained=False)

    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    assert any(name.startswith("layers.2.") for name in trainable)
    assert any(name.startswith("layers.3.") for name in trainable)
    assert any(name.startswith("layers.0.") and "ln_nir" in name for name in trainable)
    assert "patch_embed.proj.weight" not in trainable
    assert not any(name.startswith("layers.0.") and "attn" in name for name in trainable)


def test_train_step_updates_both_domains(tmp_path: Path) -> None:
    model = build_dsln_swin(MODEL_NAME, pretrained=False)
    objective = losses.ArcFaceLoss(num_classes=4, embedding_size=model.num_features)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params + list(objective.parameters()), lr=1e-3)
    empty_loader = DataLoader([])
    trainer = DualDomainTrainer(
        model, objective, optimizer, None,
        empty_loader, empty_loader, empty_loader, empty_loader, empty_loader,
        device="cpu", total_epochs=1, accumulation_steps=1, val_interval=1,
        save_dir=str(tmp_path),
    )
    norm = model.layers[3].blocks[0].norm1
    before_rgb = norm.ln_rgb.weight.clone()
    before_nir = norm.ln_nir.weight.clone()
    batch = (torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE), torch.tensor([0, 1, 2, 3]))

    loss = trainer.train_step(0, batch, batch)

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(norm.ln_rgb.weight, before_rgb)
    assert not torch.equal(norm.ln_nir.weight, before_nir)


def test_rgb_forward_gives_no_gradient_to_nir_layer_norm(swin: nn.Module) -> None:
    model = convert_to_dsln(swin)
    set_model_domain(model, "rgb")

    model(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)).sum().backward()

    norm = model.layers[3].blocks[0].norm1
    assert norm.ln_rgb.weight.grad is not None
    assert norm.ln_nir.weight.grad is None
