"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt: loss = (1 - eps) * NLL(y) + eps * mean_k(-log p_k). eps = 0 cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.eps = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1 - self.eps) * nll + self.eps * smooth).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp. gamma = 0 (alpha=None) cho đúng cross-entropy
    (kiểm tra bằng `_selftest()` ở cuối file).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
        pt = logp.exp()
        loss = -((1 - pt) ** self.gamma) * logp
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp
    """
    n = torch.as_tensor(np.asarray(counts), dtype=torch.float64)
    w = 1.0 / n if not beta else (1.0 - beta) / (1.0 - torch.pow(torch.tensor(float(beta), dtype=torch.float64), n))
    return (w / w.sum() * len(n)).float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        cut = np.sqrt(1 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def _selftest():
    """Kiểm tra nhanh: focal(gamma=0) == CE, label smoothing(eps=0) == CE, CutMix lam hợp lệ."""
    torch.manual_seed(0)
    z, y = torch.randn(32, 9), torch.randint(0, 9, (32,))
    ce = F.cross_entropy(z, y)
    assert abs(FocalLoss(0.0)(z, y) - ce) < 1e-6
    assert abs(LabelSmoothingCE(0.0)(z, y) - ce) < 1e-6
    xm, (ya, yb, lam) = mix_batch(torch.randn(8, 3, 32, 32), torch.randint(0, 9, (8,)), 1.0, "cutmix")
    assert 0.0 <= lam <= 1.0 and xm.shape == (8, 3, 32, 32)
    print("losses selftest OK")


if __name__ == "__main__":
    _selftest()
