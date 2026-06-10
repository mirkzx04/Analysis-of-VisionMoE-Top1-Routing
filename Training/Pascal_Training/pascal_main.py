import os

import torch

from torchvision.datasets import VOCSegmentation
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from Model.PCE import PCENetwork
from Pascal_Training.PascalLitModule import PascalLitModule

# Pesi caricati pre-addestrati -> usa le stesse statistiche ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VOC_IGNORE_INDEX = 255  # bordo "void" di VOC -> ignore_index nella loss


class WrapMask:
    """Converte la maschera PIL ('P') in tv_tensors.Mask long, cosi' le v2
    geometriche la trattano col nearest e le fotometriche la saltano."""
    def __call__(self, img, target):
        mask = TF.pil_to_tensor(target).squeeze(0).long()  # [H, W]
        return img, tv_tensors.Mask(mask)


def build_transforms(train, crop_size=224):
    if train:
        geom_photo = [
            # Scale jitter 0.5-2.0: resize del lato corto a una taglia casuale.
            v2.RandomResize(
                min_size=int(0.5 * crop_size),
                max_size=int(2.0 * crop_size),
                antialias=True,
            ),
            # Crop fisso; padding con 0 sull'immagine e 255 (ignore) sulla maschera.
            v2.RandomCrop(
                crop_size,
                pad_if_needed=True,
                fill={tv_tensors.Image: 0, tv_tensors.Mask: VOC_IGNORE_INDEX},
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ]
    else:
        geom_photo = [
            v2.Resize(crop_size, antialias=True),       # lato corto -> 224
            v2.CenterCrop(crop_size),
        ]

    return v2.Compose([
        WrapMask(),
        v2.ToImage(),                                   # PIL img -> tv_tensors.Image
        *geom_photo,
        v2.ToDtype(torch.float32, scale=True),          # [0,1], solo le Image
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def download_pascal_voc():
    train_set = VOCSegmentation(
        root="./data",
        year= "2012",
        image_set="train",
        download=True,
        transforms=build_transforms(train=True),
    )

    val_set = VOCSegmentation(
        root = "./data",
        year = "2012",
        image_set= "val",
        download= True,
        transforms=build_transforms(train=False),
    )

    return train_set, val_set

def instance_model(temp_init, model_path = ""):
    pce = PCENetwork(
        num_experts = 16,
        layer_number = 8,
        patch_size = 16,
        num_classes=21,
        router_temp=temp_init,
        capacity_factor_train = 2.00,
        capacity_factor_val = 2.00,
        halo_for_patches=2,
        use_static_map=False,
        unified_router = True,
        task="seg",
        uniform_epochs=0,
    )

    return pce

def clean_state_dict(state_dict):
    """Align Lightning checkpoint keys with PCENetwork parameter names."""
    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.removeprefix("model.")

        if clean_k.endswith(".gamma"):
            continue

        clean_k = clean_k.replace(".router_gate.expert_emb", ".router_gate.W")
        cleaned[clean_k] = v

    return cleaned

def load_model(temp_init, model_path):
    """
    Load checkpoint weights into PCENetwork while EXCLUDING the router and the
    prediction head, which must be retrained from scratch on Pascal VOC.

    The router trainable parameters live in `layer.router_gate`
    (RouterGate: `W` for unified, `semantic_w`/`position_w` for dual-branch);
    `self.router` has no weights. The classifier head is `prediction_head.*`.
    Filtering out these keys leaves those modules at their initialization values.
    """
    model = instance_model(temp_init)

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict(state_dict)

    # Keys to exclude: router gates and prediction head. They keep their
    # random initialization and get retrained from scratch.
    def is_excluded(k):
        return (
            "router_gate" in k
            or k.startswith("router.")
            or k.startswith("prediction_head.")
        )

    excluded_keys = [k for k in state_dict if is_excluded(k)]
    for k in excluded_keys:
        del state_dict[k]

    # strict=False because the excluded keys are intentionally missing.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Sanity check: the only missing keys must be the ones we excluded.
    unexpected_missing = [k for k in missing if not is_excluded(k)]
    if unexpected_missing:
        raise RuntimeError(f"Chiavi mancanti non previste: {unexpected_missing}")
    if unexpected:
        raise RuntimeError(f"Chiavi inattese nel checkpoint: {unexpected}")

    print(f"[load_model] Caricati {len(state_dict)} tensori; esclusi "
          f"{len(excluded_keys)} parametri (router + prediction head), "
          f"riaddestrati da zero.")

    return model

def get_accelerator_and_precision():
    if not torch.cuda.is_available():
        return "cpu", "32-true"

    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    cuda_arches = getattr(torch._C, "_cuda_getArchFlags", lambda: "")()

    print(f"-- CUDA GPU: {device_name} | compute capability: {major}.{minor} ---")
    print(f"-- PyTorch CUDA: {torch.version.cuda} | compiled arches: {cuda_arches or 'unknown'} ---")

    if torch.cuda.is_bf16_supported():
        return "cuda", "bf16-mixed"

    if major >= 7:
        print("-- BF16 not supported on this GPU: using precision='16-mixed' ---")
        return "cuda", "16-mixed"

    print("-- Mixed precision not enabled for this GPU: using precision='32-true' ---")
    return "cuda", "32-true"


if __name__ == "__main__":
    device, precision = get_accelerator_and_precision()
    print(f"-- Start with device : {device} ---")
    print(f"-- Trainer precision : {precision} ---")
    print("\n ------------------------ \n")

    # Hyperparameters
    num_classes = 21          # VOC: 20 object classes + background
    train_epochs = 60

    backbone_lr = 2e-5        # pretrained backbone: small LR
    head_lr = 1e-3            # segmentation head: trained from scratch
    router_lr = 1e-3          # router: trained from scratch
    weight_decay = 1e-3       # match Tiny_Training/main.py

    # Single shared temp_init so load_model and the LitModule agree.
    temp_init = 1.75
    temp_final = 0.50
    temp_epochs = 25

    # Data
    train_set, val_set = download_pascal_voc()
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)

    # Model: backbone/experts loaded from the unified-router checkpoint;
    # router + prediction head are reinitialised and retrained from scratch.
    model = load_model(temp_init=temp_init, model_path="checkpoints_unified_router/last.ckpt")

    lit_module = PascalLitModule(
        model=model,
        backbone_lr=backbone_lr,
        router_lr=router_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        num_classes=num_classes,
        train_epochs=train_epochs,
        temp_init=temp_init,
        temp_final=temp_final,
        temp_epochs=temp_epochs,
    )

    # Logger (same as Tiny_Training/main.py).
    logger = WandbLogger(
        project="PCE",
        log_model=False,
        name="Test-Pascal-VOC2012-Segmentation",
    )
    logger.experiment.define_metric("epoch")
    logger.experiment.define_metric("*", step_metric="epoch")

    checkpoint_callback = ModelCheckpoint(
        monitor="validation/mIoU",
        mode="max",
        save_last=True,
        filename="best-model",
        dirpath="checkpoints_pascal/",
        save_weights_only=False,
    )

    trainer = pl.Trainer(
        max_epochs=train_epochs,
        logger=logger,
        precision=precision,
        accelerator=device,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=2,
    )

    print("--- Start training --- \n")
    if os.path.exists("checkpoints_pascal/last.ckpt"):
        trainer.fit(lit_module, train_loader, val_loader, ckpt_path="checkpoints_pascal/last.ckpt")
    else:
        trainer.fit(lit_module, train_loader, val_loader)

    logger.experiment.finish()
