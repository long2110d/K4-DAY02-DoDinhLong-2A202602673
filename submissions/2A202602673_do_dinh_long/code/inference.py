"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Trả về (filenames, y_true[N], logits[N, 9]) theo đúng thứ tự loader. Chạy FP32 để tái lập
    chính xác (muốn AMP thì bọc ngoài bằng torch.autocast).
    """
    model.eval()
    names, ys, outs = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            if view is not None:
                x = view(x)
            outs.append(model(x).float().cpu().numpy())
            ys.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
            names += list(f)
    return names, np.concatenate(ys), np.concatenate(outs)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) bằng torch.flip trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[3])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch.

    Mỗi batch trong list là một view; dùng bằng cách gọi predict_logits với view=lambda x: views_multicrop(x, c)[i].
    """
    h, w = x.shape[-2:]
    assert crop <= min(h, w), f"crop {crop} lớn hơn ảnh {h}x{w}"
    t, l = (h - crop) // 2, (w - crop) // 2
    crops = [x[..., :crop, :crop], x[..., :crop, w - crop:], x[..., h - crop:, :crop],
             x[..., h - crop:, w - crop:], x[..., t:t + crop, l:l + crop]]
    if flip:
        crops += [torch.flip(c, dims=[3]) for c in crops]
    return crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ nên KHÔNG dùng hàm này cho chúng). Dùng bilinear.
    """
    return [x if x.shape[-1] == s else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Trả về xác suất (N, 9) đã chuẩn hoá (numpy float64).
    """
    zs = [torch.as_tensor(z, dtype=torch.float64) for z in logits_per_view]
    if space == "prob":
        p = torch.stack([F.softmax(z, 1) for z in zs]).mean(0)
    elif space == "logit":
        p = F.softmax(torch.stack(zs).mean(0), 1)
    else:
        raise ValueError(f"space không hợp lệ: {space}")
    return p.numpy()


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed).

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    return np.mean([np.asarray(p, dtype=np.float64) for p in list_of_probs], axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Tối ưu log T bằng LBFGS (T = exp(log T) > 0). Accuracy không đổi vì thứ tự lớp không đổi.
    KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(val_logits, dtype=torch.float64)
    y = torch.as_tensor(val_labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T) dưới dạng numpy float64."""
    return F.softmax(torch.as_tensor(logits, dtype=torch.float64) / T, dim=1).numpy()


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Gộp trên MỘT BẢN SAO (model gốc không đổi). Chỉ gộp các cặp Conv2d -> nn.BatchNorm2d liền kề
    trong cùng một module cha; BatchNormAct2d của timm (có activation bên trong) không được gộp.
    In sai số lớn nhất giữa đầu ra trước/sau gộp. Kiến trúc không có BN (ViT, Swin, ConvNeXt dùng
    LayerNorm) thì không có gì để gộp (n_fused = 0).
    """
    model.eval()
    fused = copy.deepcopy(model).eval()
    n_fused = 0
    for parent in fused.modules():
        names = list(parent._modules.keys())
        for a, b in zip(names[:-1], names[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if isinstance(conv, nn.Conv2d) and type(bn) is nn.BatchNorm2d and conv.out_channels == bn.num_features:
                scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                                conv.padding, conv.dilation, conv.groups, bias=True).to(conv.weight.device)
                with torch.no_grad():
                    new.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
                    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
                    new.bias.copy_(bn.bias + (bias - bn.running_mean) * scale)
                parent._modules[a], parent._modules[b] = new, nn.Identity()
                n_fused += 1
    device = next(model.parameters()).device
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.inference_mode():
        err = (model(x) - fused(x)).abs().max().item()
    print(f"fuse_conv_bn: đã gộp {n_fused} cặp Conv-BN, sai số lớn nhất = {err:.2e}")
    return fused
