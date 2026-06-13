"""
CelebAMask-HQ face parsing dataset (semantic segmentation, 19 classes).

Source: HuggingFace `v-xchen-v/celebamask_hq` (no auth). The dataset ships as 6
parquet shards under `data/train-{00000..00005}-of-00006.parquet` (30000 images,
~2.9GB). Each row has:
    - `image`: RGB 512x512 PNG bytes
    - `label`: single-channel index map, values 0..18 (19 classes).

There is only a `train` split (30000 images). We make a DETERMINISTIC train/val
split by index (NO shuffling): the last 1500 images go to val, the first 28500 to
train.

The parquet tables are held in memory (the PNG bytes are compressed) and decoded
lazily in `__getitem__`. Transforms are applied inside the Dataset, exactly like
Pascal's `build_transforms` (torchvision v2 with tv_tensors so masks get
nearest-neighbour and skip photometric ops).
"""

import io

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF


# --------------------------------------------------------------------------- #
# Documented id2label (v-xchen-v/celebamask_hq id2label.json)
# --------------------------------------------------------------------------- #
CELEBA_CLASS_NAMES = [
    "background", "hair", "hat", "ear_r", "skin", "l_brow", "r_brow",
    "l_eye", "r_eye", "eye_g", "nose", "u_lip", "l_lip", "mouth", "cloth",
    "neck", "neck_l", "l_ear", "r_ear",
]
CELEBA_NUM_CLASSES = 19

# Same ImageNet stats and ignore index convention as Pascal training.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CELEBA_IGNORE_INDEX = 255  # CelebA has no native void: RandomCrop padding -> 255

HF_REPO_ID = "v-xchen-v/celebamask_hq"
HF_NUM_SHARDS = 6
HF_CACHE_DIR = "data/_hf_cache"

# Deterministic split sizes.
VAL_SIZE = 1500  # last 1500 images -> val; the rest -> train


# --------------------------------------------------------------------------- #
# Transforms: mirror `build_transforms` in pascal_main EXACTLY.
# --------------------------------------------------------------------------- #
class WrapMask:
    """Convert the PIL/index-map mask into a tv_tensors.Mask (long) so the v2
    geometric ops treat it with NEAREST and the photometric ops skip it."""
    def __call__(self, img, target):
        mask = TF.pil_to_tensor(target).squeeze(0).long()  # [H, W]
        return img, tv_tensors.Mask(mask)


def build_transforms(train, crop_size=224):
    if train:
        geom_photo = [
            # Scale jitter 0.5-2.0: resize the short side to a random size.
            v2.RandomResize(
                min_size=int(0.5 * crop_size),
                max_size=int(2.0 * crop_size),
                antialias=True,
            ),
            # Fixed crop; pad image with 0 and mask with 255 (ignore).
            v2.RandomCrop(
                crop_size,
                pad_if_needed=True,
                fill={tv_tensors.Image: 0, tv_tensors.Mask: CELEBA_IGNORE_INDEX},
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ]
    else:
        geom_photo = [
            v2.Resize(crop_size, antialias=True),       # short side -> 224
            v2.CenterCrop(crop_size),
        ]

    return v2.Compose([
        WrapMask(),
        v2.ToImage(),                                   # PIL img -> tv_tensors.Image
        *geom_photo,
        v2.ToDtype(torch.float32, scale=True),          # [0,1], only the Image
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Parquet loading helpers
# --------------------------------------------------------------------------- #
def _decode_img(cell):
    """Decode a parquet image/label cell (PNG bytes) into a PIL.Image.

    Mirrors `_decode_img` in Testing/experiments/mi_pos_class.py but returns a
    PIL.Image (not a numpy array) so the v2 transform pipeline can consume it.
    """
    if isinstance(cell, dict) and cell.get("bytes") is not None:
        return Image.open(io.BytesIO(cell["bytes"]))
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell))
    raise ValueError(f"cannot decode parquet image cell of type {type(cell)}")


def _load_shards(num_shards, cache_dir):
    """Download (if needed) and read the requested parquet shards into memory.

    Returns two parallel lists (image_cells, label_cells), in dataset order,
    each entry being the raw parquet cell (compressed PNG bytes wrapped in a
    dict/bytes). Decoding is deferred to __getitem__.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    image_cells = []
    label_cells = []
    for i in range(num_shards):
        fn = f"data/train-{i:05d}-of-00006.parquet"
        path = hf_hub_download(
            HF_REPO_ID, fn, repo_type="dataset", cache_dir=cache_dir
        )
        table = pq.read_table(path, columns=["image", "label"])
        img_col = table.column("image")
        lab_col = table.column("label")
        for j in range(len(img_col)):
            image_cells.append(img_col[j].as_py())
            label_cells.append(lab_col[j].as_py())

    return image_cells, label_cells


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CelebAMaskHQDataset(Dataset):
    """CelebAMask-HQ face-parsing dataset for one split (train or val).

    Holds the (compressed) parquet cells in memory and decodes lazily. Applies
    a `transforms` callable (Pascal-style: takes (PIL image, PIL/index mask),
    returns (image_tensor, mask_long_tensor)).
    """

    def __init__(self, image_cells, label_cells, transforms):
        super().__init__()
        assert len(image_cells) == len(label_cells)
        self.image_cells = image_cells
        self.label_cells = label_cells
        self.transforms = transforms

    def __len__(self):
        return len(self.image_cells)

    def __getitem__(self, idx):
        img = _decode_img(self.image_cells[idx]).convert("RGB")
        mask = _decode_img(self.label_cells[idx])
        # Label maps are single-channel index images; force mode 'L' so a single
        # 0..18 channel is read (some PNGs may decode as 'P'/'I').
        if mask.mode not in ("L", "I"):
            mask = mask.convert("L")

        img_t, mask_t = self.transforms(img, mask)
        # mask_t is a tv_tensors.Mask (long, [H, W]); return a plain long tensor.
        return img_t, mask_t.long()


def build_celeba_datasets(num_shards=HF_NUM_SHARDS, cache_dir=HF_CACHE_DIR,
                          crop_size=224, val_size=VAL_SIZE):
    """Build the deterministic (train, val) CelebAMask-HQ dataset pair.

    Args:
        num_shards : how many of the 6 parquet shards to load (use 1 for smoke
                     tests so you don't download the full ~2.9GB). The default
                     loads all 6 for real training.
        cache_dir  : HuggingFace cache dir.
        crop_size  : square crop / resize size (224 to match the model input).
        val_size   : number of trailing images held out for validation.

    Split is deterministic by index: the LAST `val_size` images go to val, the
    rest to train. No shuffling in the split itself.
    """
    image_cells, label_cells = _load_shards(num_shards, cache_dir)
    n = len(image_cells)

    # Deterministic split: last `val_size` -> val, first (n - val_size) -> train.
    n_val = min(val_size, n)
    split = n - n_val

    train_imgs, train_labs = image_cells[:split], label_cells[:split]
    val_imgs, val_labs = image_cells[split:], label_cells[split:]

    train_set = CelebAMaskHQDataset(
        train_imgs, train_labs, transforms=build_transforms(train=True, crop_size=crop_size)
    )
    val_set = CelebAMaskHQDataset(
        val_imgs, val_labs, transforms=build_transforms(train=False, crop_size=crop_size)
    )

    return train_set, val_set
