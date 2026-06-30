import os
import sys
from pathlib import Path

# Rendi lo script eseguibile direttamente: aggiungi la root del progetto (per
# `Model`, `Datasets_Classes`, ...) e la cartella `Training/` (per `train_utils`,
# `schedulers`, il package `Pascal_Training`) al sys.path.
_ROOT = Path(__file__).resolve().parents[2]       # .../Positional_Convolution_Experts-main
_TRAINING = Path(__file__).resolve().parents[1]   # .../Training
for _p in (_ROOT, _TRAINING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
from config import HParams, RouterTypesParams, LossWeights, EpochParams

# Normalizzazione con statistiche ImageNet (standard).
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

def instance_model(hp: HParams, rt: RouterTypesParams, ep: EpochParams):
    pce = PCENetwork(
        num_experts = hp.num_experts,
        layer_number = hp.layer_number,
        patch_size = hp.patch_size,
        num_classes=hp.num_classes,
        router_temp=hp.temp_init,
        capacity_factor_train = hp.capacity_factor_train,
        capacity_factor_val = hp.capacity_factor_val,
        halo_for_patches=hp.halo_for_patches,
        input_size=hp.input_size,
        use_static_map=rt.use_static_map,
        pos_only=rt.pos_only,
        semantic_only=rt.semantic_only,
        unified_router=rt.unified_router,
        interaction=rt.interaction,
        interaction_hidden_size=rt.interaction_hidden_size,
        interaction_include_main_effects=rt.interaction_include_main_effects,
        task=hp.task,
        uniform_epochs=ep.uniform_epochs,
    )

    return pce

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

    # Hyperparameters (VOC segmentation, trained from scratch).
    hp = HParams(
        # model
        num_experts=16,
        layer_number=8,
        patch_size=16,
        halo_for_patches=2,
        input_size=224,
        capacity_factor_train=2.00,
        capacity_factor_val=2.00,
        task="seg",
        num_classes=21,            # VOC: 20 object classes + background
        # optim / train
        batch_size=64,
        weight_decay=1e-2,         # match Tiny_Training/main.py
        backbone_lr=1e-3,          # from scratch
        head_lr=1e-3,              # segmentation head: trained from scratch
        router_lr=5e-4,            # router: trained from scratch
        accumulate_grad_batches=2,
        adam_betas=(0.9, 0.98),
        adam_eps=1e-8,
        temp_init=1.5,             # allineato a CelebA (era 1.75)
        temp_final=0.50,
        # clipping
        backbone_max_norm=1.5,
        router_max_norm=0.5,
        head_max_norm=1.0,
    )
    rt = RouterTypesParams(
        use_static_map=False,
        pos_only=False,
        semantic_only=False,
        interaction=True,
        unified_router=False,
        interaction_hidden_size=128,
        interaction_include_main_effects=True,   # allineato a CelebA (era False)
    )
    # Epoch counts / warmups (schedule durations).
    ep = EpochParams(
        train_epochs=100,          # allineato a CelebA (era 60)
        uniform_epochs=10,
        warmup_backbone=5,
        head_warmup=5,
        router_warmup=5,
        temp_epochs=10,
    )
    lw = LossWeights(
        z_loss_weight=1e-4,        # allineato a CelebA (era 1e-2)
        label_smoothing=0.0,
        ignore_index=VOC_IGNORE_INDEX,
    )

    # Data
    train_set, val_set = download_pascal_voc()
    train_loader = DataLoader(train_set, batch_size=hp.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=hp.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model: from scratch (fresh weights, no checkpoint).
    model = instance_model(hp, rt, ep)

    lit_module = PascalLitModule(
        model=model,
        hparams=hp,
        loss_weights=lw,
        epoch_params=ep,
        static="learnable",        # allineato a CelebA (era "not_learnable"; inerte con interaction gate)
        router_types=rt,
    )

    # Logger (same as Tiny_Training/main.py).
    logger = WandbLogger(
        project="PCE",
        log_model=False,
        name="Pascal-VOC2012-Segmentation - From Scratch-Interaction",
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
        max_epochs=ep.train_epochs,
        logger=logger,
        precision=precision,
        accelerator=device,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=hp.accumulate_grad_batches,
    )

    print("--- Start training --- \n")
    if os.path.exists("checkpoints_pascal/last.ckpt"):
        trainer.fit(lit_module, train_loader, val_loader, ckpt_path="checkpoints_pascal/last.ckpt")
    else:
        trainer.fit(lit_module, train_loader, val_loader)

    logger.experiment.finish()
