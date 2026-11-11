"""SDOBenchmark loading: download, sample index, splits, torch Dataset."""

import glob
import os
import random
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

EUV_CHANNELS = ("94", "131", "171", "193", "211", "304", "335", "1700")
MAGNETOGRAM = "magnetogram"

URLS = {
    "example": "https://github.com/i4Ds/SDOBenchmark/archive/data-example.zip",
    "full": "https://github.com/i4Ds/SDOBenchmark/archive/data-full.zip",
}

_IMG_CACHE = {}


def download(variant="full", base=None):
    base = base or ("/content" if os.path.isdir("/content") else ".")
    zip_path = os.path.join(base, f"sdobenchmark-{variant}.zip")
    extract_root = os.path.join(base, f"sdo_{variant}")

    data_dir = _find_data_dir(extract_root) if os.path.isdir(extract_root) else None
    if data_dir:
        return data_dir

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(URLS[variant], zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_root)

    data_dir = _find_data_dir(extract_root)
    if data_dir is None:
        raise RuntimeError("no training/test folders after extraction")
    return data_dir


def _find_data_dir(root):
    for d, subs, _ in os.walk(root):
        if "training" in subs and "test" in subs:
            return d
    return None


def _parse_channel(path):
    stem = os.path.basename(path)
    if stem.lower().endswith(".jpg"):
        stem = stem[:-4]
    if "__" not in stem:
        return None, None
    return stem.rsplit("__", 1)


def _id_to_folder(split_dir, sample_id):
    if "_" in sample_id:
        ar, rest = sample_id.split("_", 1)
        cand = os.path.join(split_dir, ar, rest)
        if os.path.isdir(cand):
            return cand
    cand = os.path.join(split_dir, sample_id)
    return cand if os.path.isdir(cand) else None


def load_split(data_dir, split, flux_threshold=1e-6):
    """Index one split. Each sample keeps the latest timestep per channel."""
    split_dir = os.path.join(data_dir, split)
    meta = pd.read_csv(os.path.join(split_dir, "meta_data.csv"))

    samples = []
    for _, row in meta.iterrows():
        sample_id = str(row["id"]).strip().strip("/")
        folder = _id_to_folder(split_dir, sample_id)
        if folder is None:
            continue

        latest = {}
        for path in glob.glob(os.path.join(folder, "*.jpg")):
            ts, channel = _parse_channel(path)
            if channel is None:
                continue
            if channel not in latest or ts > latest[channel][0]:
                latest[channel] = (ts, path)

        channels = {c: p for c, (_, p) in latest.items()}
        if MAGNETOGRAM not in channels:
            continue

        try:
            peak = float(row["peak_flux"])
        except (TypeError, ValueError):
            continue

        samples.append({
            "id": sample_id,
            "folder": folder,
            "channels": channels,
            "peak_flux": peak,
            "label": int(peak >= flux_threshold),
        })
    return samples


def class_balance(samples):
    y = np.array([s["label"] for s in samples])
    return len(samples), int(y.sum()), float(y.mean()) if len(y) else 0.0


def pos_weight(samples):
    y = np.array([s["label"] for s in samples])
    return float((len(y) - y.sum()) / max(y.sum(), 1))


def stratified_split(samples, val_frac, seed=42):
    rng = random.Random(seed)
    pos = [s for s in samples if s["label"] == 1]
    neg = [s for s in samples if s["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos = max(1, int(len(pos) * val_frac))
    n_neg = max(1, int(len(neg) * val_frac))
    val = pos[:n_pos] + neg[:n_neg]
    train = pos[n_pos:] + neg[n_neg:]
    rng.shuffle(val)
    rng.shuffle(train)
    return train, val


def subsample(samples, max_n=0, neg_per_pos=0.0, seed=42):
    if not max_n and not neg_per_pos:
        return samples
    rng = random.Random(seed)
    pos = [s for s in samples if s["label"] == 1]
    neg = [s for s in samples if s["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if neg_per_pos:
        neg = neg[: int(len(pos) * neg_per_pos)]
    out = pos + neg
    rng.shuffle(out)
    return out[:max_n] if max_n else out


def load_gray(path, img_size):
    cached = _IMG_CACHE.get(path)
    if cached is not None:
        return cached
    img = Image.open(path).convert("L").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    _IMG_CACHE[path] = arr
    return arr


# Feeds samples to the training loop. Each item comes back as three things: the
# magnetogram (1 x H x W), the EUV passbands stacked into one tensor (C x H x W),
# and the label. Channels that are missing from a sample are filled with zeros,
# which happens often enough in SDOBenchmark to matter. With train=True it also
# flips images left-right half the time.
class SDOSet(Dataset):
    def __init__(self, samples, img_size=64, euv_channels=EUV_CHANNELS, train=False):
        self.samples = samples
        self.img_size = img_size
        self.euv_channels = list(euv_channels)
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        channels = self.samples[i]["channels"]
        zero = np.zeros((self.img_size, self.img_size), np.float32)

        mag = (load_gray(channels[MAGNETOGRAM], self.img_size)
               if MAGNETOGRAM in channels else zero)[None]
        euv = np.stack([
            load_gray(channels[c], self.img_size) if c in channels else zero
            for c in self.euv_channels
        ])

        if self.train and random.random() < 0.5:
            mag = mag[:, :, ::-1].copy()
            euv = euv[:, :, ::-1].copy()

        return (torch.from_numpy(mag),
                torch.from_numpy(euv),
                torch.tensor(float(self.samples[i]["label"])))
