"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Một hàm `run(cfg)` dùng cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) được tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm.
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
import typing
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset as ds
import losses
import model as mdl


def _import_eval():
    """Nạp eval.py gốc của repo (không sửa): tìm trong sys.path rồi các thư mục cha của file này."""
    try:
        import eval as ev
        if hasattr(ev, "compute_metrics"):
            return ev
    except ImportError:
        pass
    for p in Path(__file__).resolve().parents:
        if (p / "eval.py").is_file():
            sys.path.insert(0, str(p))
            sys.modules.pop("eval", None)
            import eval as ev
            return ev
    raise ImportError("không tìm thấy eval.py của repo gốc")


ev = _import_eval()


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    Đặt seed random/numpy/torch (CPU+CUDA), cudnn.deterministic=True, benchmark=False. Chưa bật
    use_deterministic_algorithms, nên một vài phép CUDA vẫn có thể lệch nhẹ giữa hai lần chạy cùng
    seed. Worker của DataLoader được seed trong dataset.make_loader.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = mdl.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55).

    Cập nhật THEO BƯỚC (iteration): hệ số nhân LR = (t+1)/warmup khi t < warmup, sau đó
    0.5 * (1 + cos(pi * tiến độ)) về 0. Hệ số nhân áp lên LR gốc của từng nhóm tham số.
    """
    total = max(1, cfg.epochs * steps_per_epoch)
    warm = int(cfg.warmup_epochs * steps_per_epoch)

    def factor(step):
        if warm > 0 and step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Giữ một bản sao `self.module` để đánh giá. Tham số được làm trung bình động; buffer
    (BatchNorm running_mean/var, num_batches_tracked) được sao chép thẳng từ model đang train.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.module.buffers(), model.buffers()):
            e.copy_(b)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict {"train_loss": ..., "lr": ...}.

    model.train() (backbone đóng băng thì giữ ở eval, xem model.set_train_mode); AMP (autocast +
    GradScaler); clip gradient 1.0; scheduler.step() sau mỗi bước; cập nhật EMA nếu có.
    """
    mdl.set_train_mode(model)
    use_amp = cfg.amp and device.type == "cuda"
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if cfg.mix:
                x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
                loss = losses.mixed_loss(criterion, model(x), targets)
            else:
                loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return {"train_loss": total / max(1, n), "lr": optimizer.param_groups[-1]["lr"]}


def evaluate(model, loader, criterion, device, amp: bool = False):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file. Loss là trung bình theo số mẫu trên
    TOÀN BỘ tập (không phải trung bình các batch).
    """
    model.eval()
    names, ys, outs, total = [], [], [], 0.0
    with torch.inference_mode():
        for x, y, f in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
                z = model(x)
            z = z.float()
            total += criterion(z, y).item() * x.size(0)
            names += list(f)
            ys.append(y.cpu().numpy())
            outs.append(z.cpu().numpy())
    y_true, logits = np.concatenate(ys), np.concatenate(outs)
    return names, y_true, logits, total / len(y_true)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    Ba ô: loss train/val; macro-F1 và top-1 val (đánh dấu epoch tốt nhất); LR theo epoch.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    best_epoch = int(h["epoch"].iloc[int(h["val_macro_f1"].values.argmax())])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(h["epoch"], h["train_loss"], marker="o", label="train loss")
    ax[0].plot(h["epoch"], h["val_loss"], marker="o", label="val loss")
    ax[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax[1].plot(h["epoch"], h["val_macro_f1"], marker="o", label="val macro-F1")
    ax[1].plot(h["epoch"], h["val_top1"], marker="o", label="val top-1")
    ax[1].axvline(best_epoch, color="gray", ls="--", label=f"best epoch {best_epoch}")
    ax[1].set(title="Validation", xlabel="epoch", ylabel="score")
    ax[2].plot(h["epoch"], h["lr"], marker="o", label="LR (nhóm head)")
    ax[2].set(title="Learning rate", xlabel="epoch", ylabel="lr")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    Checkpoint tốt nhất chọn theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn).
    """
    set_seed(cfg.seed)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, cfg.fold)
    ds.check_split(train_df, val_df, test_df, cfg.images_dir)
    train_loader = ds.make_loader(train_df, cfg.images_dir, ds.build_transforms(True, cfg.img_size, cfg.aug),
                                  cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    eval_tf = ds.build_transforms(False, cfg.img_size)
    val_loader = ds.make_loader(val_df, cfg.images_dir, eval_tf, cfg.batch_size * 2, False,
                                num_workers=cfg.num_workers)

    model = mdl.build_model(cfg.backbone, pretrained=cfg.init != "scratch", num_classes=ds.NUM_CLASSES,
                            drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    counts = np.bincount(train_df["Label"].to_numpy(), minlength=ds.NUM_CLASSES)
    kw = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    w = None
    if cfg.class_weight_beta is not None:
        w = losses.class_weights(counts, cfg.class_weight_beta).to(device)
    if cfg.loss == "ce_weighted":
        kw["weight"] = w if w is not None else losses.class_weights(counts, 0.0).to(device)
    elif cfg.loss == "focal":
        kw["alpha"] = w
    criterion = losses.build_criterion(cfg.loss, **kw).to(device)
    val_criterion = torch.nn.CrossEntropyLoss()  # val loss luôn là CE thường để so sánh được giữa các cấu hình
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.module if ema is not None else model

    history, best_f1, best_epoch, ckpt = [], -1.0, -1, out / "best.pt"
    epoch_times = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        _, y_true, logits, vloss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp)
        epoch_times.append(time.perf_counter() - t0)
        probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
        m = ev.compute_metrics(y_true, probs.argmax(1), probs)  # macro-F1 trên TOÀN BỘ val
        row = {"epoch": epoch, **tr, "val_loss": vloss, "val_macro_f1": m["macro_f1"],
               "val_top1": m["top1"], "epoch_time_s": epoch_times[-1]}
        history.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        if m["macro_f1"] > best_f1:  # strict '>' : hòa thì giữ epoch sớm hơn
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(eval_model.state_dict(), ckpt)

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    eval_model.load_state_dict(torch.load(ckpt, map_location=device))
    names, y_true, logits, _ = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    np.save(out / "val_logits.npy", logits)
    ev.save_predictions(pred_path(cfg, "val"), names, y_true, probs)
    vm = ev.compute_metrics(y_true, probs.argmax(1), probs)

    if cfg.save_test_predictions:  # chỉ ở Bước 4: đúng MỘT lần, không dùng để chọn gì
        test_loader = ds.make_loader(test_df, cfg.images_dir, eval_tf, cfg.batch_size * 2, False,
                                     num_workers=cfg.num_workers)
        tn, ty, tl, _ = evaluate(eval_model, test_loader, val_criterion, device, cfg.amp)
        tp = torch.softmax(torch.from_numpy(tl), 1).numpy()
        np.save(out / "test_logits.npy", tl)
        ev.save_predictions(pred_path(cfg, "test"), tn, ty, tp)

    plot_curves(history, Path("curves") / f"{cfg.exp_id}_seed{cfg.seed}_curves.png",
                f"{cfg.exp_id} seed{cfg.seed} - {cfg.backbone} ({cfg.init})")
    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
               "weight_tag": getattr(model, "weight_tag", None), "best_epoch": best_epoch,
               "val_macro_f1": vm["macro_f1"], "val_top1": vm["top1"], "val_nll": vm["nll"],
               "sec_per_epoch": float(np.mean(epoch_times)),
               "params_m": mdl.count_params(model),
               "gmacs": mdl.count_gmacs(model, cfg.img_size)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    none/null -> None; bool nhận true/false/1/0; các kiểu còn lại ép theo annotation của field.
    """
    hints = typing.get_type_hints(Config)
    names = {f.name for f in fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override phải có dạng KEY=VALUE, nhận '{pair}'")
        key, val = pair.split("=", 1)
        if key not in names:
            raise KeyError(f"'{key}' không có trong Config; các key hợp lệ: {sorted(names)}")
        if val.lower() in ("none", "null"):
            out[key] = None
            continue
        t = ([a for a in typing.get_args(hints[key]) if a is not type(None)] or [hints[key]])[0]
        if t is bool:
            if val.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{key}: giá trị bool không hợp lệ '{val}'")
            out[key] = val.lower() in ("true", "1")
        else:
            out[key] = t(val)
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == "__main__":
    main()
