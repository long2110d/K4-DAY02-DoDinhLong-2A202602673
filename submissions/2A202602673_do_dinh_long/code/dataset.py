"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    """
    labels_dir = Path(labels_dir)
    return tuple(pd.read_csv(labels_dir / f"{part}_subset{fold}.csv") for part in ("train", "val", "test"))


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    TODO kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    images_dir = Path(images_dir)
    parts = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in parts.items()}
    per_class = {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).tolist()
                 for k, v in parts.items()}
    sets = {k: set(v["Filename"]) for k, v in parts.items()}
    for k, v in parts.items():
        assert len(sets[k]) == len(v), f"{k}: có Filename trùng lặp"
        assert v["Label"].between(0, NUM_CLASSES - 1).all(), f"{k}: nhãn ngoài [0, {NUM_CLASSES - 1}]"
    overlap = {"train&val": len(sets["train"] & sets["val"]),
               "train&test": len(sets["train"] & sets["test"]),
               "val&test": len(sets["val"] & sets["test"])}
    assert all(c == 0 for c in overlap.values()), f"các tập giao nhau: {overlap}"
    total = sum(n.values())
    assert total == 17509, f"tổng số ảnh {total} != 17509"
    missing = [f for k in parts for f in parts[k]["Filename"] if not (images_dir / f).is_file()]
    assert not missing, f"{len(missing)} ảnh không có trong {images_dir}, ví dụ {missing[:3]}"
    print("n:", n, "| ratio:", {k: round(v / total, 3) for k, v in n.items()})
    for k in parts:
        print(f"per_class[{k}]:", per_class[k])
    print("overlap:", overlap, "| total:", total, "| missing files: 0")
    return {"n": n, "per_class": per_class, "overlap": overlap, "total": total, "missing": 0}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    Lựa chọn của bài: val/test dùng Resize(img_size) rồi CenterCrop(img_size) (với img_size=224 là
    resize 256 -> 224 theo tỉ lệ 0.875, tức Resize(int(img_size/0.875)) + CenterCrop).
    Không lật dọc: ảnh cỏ dại chụp từ trên xuống nhưng khung cảnh có hướng (đất dưới, cây trên).
    """
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return T.Compose([T.Resize(int(round(img_size / 0.875))), T.CenterCrop(img_size),
                          T.ToTensor(), norm])
    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment())
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose(ops + [T.ToTensor(), norm])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.filenames = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        name = self.filenames[i]
        img = Image.open(self.images_dir / name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), name


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    """
    ds = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed() % 2**31)  # phụ thuộc set_seed gọi trước đó

    def _worker_init(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)

    smp, shuffle = None, train
    if train and sampler == "balanced":
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        w = 1.0 / counts[np.asarray(ds.labels)]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ds),
                                    replacement=True, generator=generator)
        shuffle = False
    elif sampler not in (None, "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=smp,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train, worker_init_fn=_worker_init, generator=generator,
                      persistent_workers=num_workers > 0)
