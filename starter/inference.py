"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    TODO: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    raise NotImplementedError("TODO")


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). TODO: dùng torch.flip trên chiều rộng (slide trang 75)."""
    raise NotImplementedError("TODO")


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. TODO."""
    raise NotImplementedError("TODO")


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. TODO.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    raise NotImplementedError("TODO")


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    TODO: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    raise NotImplementedError("TODO")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). TODO.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    raise NotImplementedError("TODO")


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    TODO: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    raise NotImplementedError("TODO")


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). TODO."""
    raise NotImplementedError("TODO")


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    TODO:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    raise NotImplementedError("TODO")
