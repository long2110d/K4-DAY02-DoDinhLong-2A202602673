"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    model = timm.create_model(name, pretrained=init != "scratch", num_classes=num_classes,
                              drop_rate=drop_rate)
    model.init_mode = init
    if init == "frozen":
        freeze_backbone(model)
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    model.weight_tag = cfg.get("tag") or cfg.get("hf_hub_id") or cfg.get("url") or "random-init"
    return model


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone bị đóng băng thì giữ phần backbone (BN, dropout) ở eval."""
    model.train()
    if getattr(model, "init_mode", None) == "frozen":
        head = {id(m) for m in model.get_classifier().modules()}
        for m in model.modules():
            if id(m) not in head and not any(True for _ in m.children()):
                m.eval()


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    decay, no_decay, head = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim > 1:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    Công cụ: torch.utils.flop_counter.FlopCounterMode (đếm conv/matmul, FLOPs = 2 x MAC) rồi chia 2.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(x)
    model.train(was_training)
    return fc.get_total_flops() / 2 / 1e9
