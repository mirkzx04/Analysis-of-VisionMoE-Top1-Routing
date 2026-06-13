import os
import sys
from pathlib import Path

# Make the script executable directly: add the project root (for `Model`,
# `Datasets_Classes`, ...) and `Training/` (for `train_utils`, `schedulers`,
# `config`, the `CelebA_Training` package) to sys.path.
_ROOT = Path(__file__).resolve().parents[2]       # .../Positional_Convolution_Experts-main
_TRAINING = Path(__file__).resolve().parents[1]   # .../Training
for _p in (_ROOT, _TRAINING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from Model.PCE import PCENetwork
from Datasets_Classes.CelebAMaskHQ import build_celeba_datasets
from CelebA_Training.CelebaLitModule import CelebaLitModule
from config import HParams, LossWeights, EpochParams


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


def instance_model(hp: HParams, ep: EpochParams):
    """Instantiate PCENetwork FROM SCRATCH (fresh random weights, NO checkpoint)."""
    pce = PCENetwork(
        num_experts=hp.num_experts,
        layer_number=hp.layer_number,
        patch_size=hp.patch_size,
        num_classes=hp.num_classes,
        router_temp=hp.temp_init,
        capacity_factor_train=hp.capacity_factor_train,
        capacity_factor_val=hp.capacity_factor_val,
        halo_for_patches=hp.halo_for_patches,
        input_size=hp.input_size,
        use_static_map=hp.use_static_map,
        unified_router=hp.unified_router,
        task="seg",
        uniform_epochs=ep.uniform_epochs,
    )
    return pce


if __name__ == "__main__":
    device, precision = get_accelerator_and_precision()
    print(f"-- Start with device : {device} ---")
    print(f"-- Trainer precision : {precision} ---")
    print("\n ------------------------ \n")

    # CelebAMask-HQ face parsing, from scratch (19 classes).
    hparams = HParams(
        num_experts=16,
        layer_number=8,

        patch_size=16,
        halo_for_patches=2,
        input_size=224,
        capacity_factor_train=2.0,
        capacity_factor_val=2.0,
        use_static_map=False,
        unified_router=False,
        task="seg",
        num_classes=19,
        batch_size=64,
        weight_decay=1e-2,
        backbone_lr=1e-3,   # from scratch (NOT the tiny pretrained 2e-5)
        head_lr=1e-3,
        router_lr=1e-4,
        accumulate_grad_batches=2,
        adam_betas=(0.9, 0.98),
        adam_eps=1e-8,
        temp_init=1.5,
        temp_mid=1.2,
        temp_final=0.5,
        backbone_max_norm=1.5,
        router_max_norm=0.5,
        head_max_norm=1.0,
    )
    # Epoch counts / warmups (schedule durations).
    ep = EpochParams(
        train_epochs=100,
        uniform_epochs=10,
        warmup_backbone=5,
        head_warmup=5,
        router_warmup=5,
        temp_epochs=10,
    )

    loss_weights = LossWeights(
        z_loss_weight=1e-2,
        spatial_loss_weight=1e-4,
        label_smoothing=0.0,
        ignore_index=255,
    )

    # Data: CelebAMask-HQ, deterministic index split (last 1500 -> val).
    train_set, val_set = build_celeba_datasets(crop_size=hparams.input_size)
    train_loader = DataLoader(
        train_set, batch_size=hparams.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=hparams.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"--- Dataset loaded: {len(train_set)} train / {len(val_set)} val --- \n")

    # Model: FROM SCRATCH (fresh random weights, NO checkpoint).
    model = instance_model(hparams, ep)

    lit_module = CelebaLitModule(
        model=model,
        hparams=hparams,
        loss_weights=loss_weights,
        epoch_params=ep,
        static="learnable",
    )

    # Logger (same style as the other trainings).
    logger = WandbLogger(
        project="PCE",
        log_model=False,
        name="CelebAMask-HQ Face Parsing - From Scratch - Pos. Only",
    )
    logger.experiment.define_metric("epoch")
    logger.experiment.define_metric("*", step_metric="epoch")

    checkpoint_callback = ModelCheckpoint(
        monitor="validation/mIoU",
        mode="max",
        save_last=True,
        filename="best-model",
        dirpath="checkpoints_celeba/",
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
        accumulate_grad_batches=hparams.accumulate_grad_batches,
    )

    print("--- Start training --- \n")
    if os.path.exists("checkpoints_celeba/last.ckpt"):
        trainer.fit(lit_module, train_loader, val_loader, ckpt_path="checkpoints_celeba/last.ckpt")
    else:
        trainer.fit(lit_module, train_loader, val_loader)

    logger.experiment.finish()
