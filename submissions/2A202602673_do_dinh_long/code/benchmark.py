"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không: ở đây KHÔNG tính (chỉ đo forward của model)
"""
from __future__ import annotations

import copy
import platform
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    Trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}.
    """
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ts)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Chỉ đo forward, tensor đã nằm sẵn trên thiết bị: KHÔNG gồm đọc ảnh, tiền xử lý hay chép
    CPU->GPU (preprocessing_included=False). dtype: "fp32" | "amp" (autocast) | "fp16" (model.half()).
    Ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định.
    Model gốc không bị đổi (đo trên bản sao).
    """
    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError(f"dtype không hợp lệ: {dtype}")
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    dev = device if use_cuda else "cpu"
    m = copy.deepcopy(model).to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        m, x = m.half(), x.half()
    sync = torch.cuda.synchronize if use_cuda else None

    def fn():
        with torch.inference_mode(), torch.autocast(device_type="cuda" if use_cuda else "cpu",
                                                    dtype=torch.float16, enabled=dtype == "amp" and use_cuda):
            m(x)

    r = bench(fn, warmup, iters, sync)
    gpu = torch.cuda.get_device_name(0) if use_cuda else f"CPU ({platform.processor() or platform.machine()})"
    return {"gpu": gpu, "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "preprocessing_included": False, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "images_per_s": batch_size / (r["p50"] / 1000), "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view, đo thật (K view gộp thành một batch K*batch_size), kèm so sánh với
    xấp xỉ K * p50 của một view (slide trang 63).

    `batch_size` trong kw là batch của MỘT view (mặc định 1).
    """
    kw = dict(kw)
    bs = kw.pop("batch_size", 1)
    one = latency_report(model, batch_size=bs, **kw)
    k = latency_report(model, batch_size=bs * k_views, **kw)
    return {**k, "k_views": k_views, "p50_one_view": one["p50"], "k_times_p50": k_views * one["p50"],
            "ratio_vs_k_p50": k["p50"] / (k_views * one["p50"])}
