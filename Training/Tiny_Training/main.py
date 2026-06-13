import os
import sys
from pathlib import Path
import numpy as np
import json

# Make the script executable directly by adding the project root (for `Model`,
# `Datasets_Classes`, ...) and `Training/` (for `Tiny_Training`, `schedulers`,
# `train_utils`) to sys.path.
_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _TRAINING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
gc.collect()
torch.cuda.empty_cache()

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from Datasets_Classes.TinyImageNet import TinyImageNetDataset, TinyImageNetTrainDataset, TinyImageNetValidationDataset
from Model.PCE import PCENetwork

# from Datasets_Classes.PascalVOC import PascalVOCDataset, PascalVOCTrainDataset, PascalVOCValidationDataset
# from Datasets_Classes.PatchExtractor import PatchExtractor

from Tiny_Training.TinyLitModule import TinyLitModule
from config import HParams, LossWeights, AugParams, EpochParams

def count_number_of_classes(train_labels, val_labels):
    return len(np.unique(np.concatenate([train_labels, val_labels])).tolist())

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

def get_tinyimagenet_sets(batch_size, tinyimagenet_path = 'Data/tiny-imagenet-200'):
    """
    Initialize the Tiny-ImageNet dataset and create Dataloaders for training and validation

    Args:
        tinyimagenet_path : Path of the Tiny-ImageNet dataset
    """

    tiny_image_net = TinyImageNetDataset(tinyimagenet_path)
    tiny_image_net_train_set = TinyImageNetTrainDataset(tiny_image_net)
    tiny_image_net_val_set = TinyImageNetValidationDataset(tiny_image_net)

    tiny_image_net_train_dataloader = DataLoader(dataset=tiny_image_net_train_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    tiny_image_net_val_dataloader = DataLoader(dataset=tiny_image_net_val_set, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    # Get numer of classes in labels
    all_labeles = np.concatenate([tiny_image_net_train_set.labels, tiny_image_net_val_set.labels])
    unique_labels = np.unique(all_labeles).tolist()
    num_classes = len(unique_labels)

    tiny_image_net_dic = {
        'datasets' : {
            'train' : tiny_image_net_train_set,
            'val': tiny_image_net_val_set
        },
        'dataloaders' : {
            'train' : tiny_image_net_train_dataloader,
            'val' : tiny_image_net_val_dataloader
        },
        'name' : 'TinyImage-Net',
        'num_classes' : num_classes,
        'unique_lables' : unique_labels
    }

    with open('class_mapping', 'w') as f:
        json.dump(tiny_image_net.class_to_idx, f, indent=4)

    return tiny_image_net_dic

if __name__ == "__main__":

    train_datasets = []

    device, precision = get_accelerator_and_precision()
    print(f'-- Start with device : {device} ---')
    print(f'-- Trainer precision : {precision} ---')
    print('\n ------------------------ \n')

    # Hyperparameters (Tiny-ImageNet classification). num_classes is set after
    # the dataset is loaded. Tiny historically called the backbone LR `lr`
    # (lr=0.001) -> mapped to backbone_lr; router_start_epoch == uniform_epochs.
    hp = HParams(
        # model
        num_experts=16,
        layer_number=8,
        patch_size=16,
        halo_for_patches=2,
        input_size=224,
        capacity_factor_train=2.00,
        capacity_factor_val=2.00,
        use_static_map=False,
        unified_router=False,
        task="class",
        # optim / train
        batch_size=64,
        weight_decay=1e-3,         # M
        backbone_lr=0.001,         # Tiny's `lr`
        router_lr=1e-4,
        accumulate_grad_batches=2,
        adam_betas=(0.9, 0.98),
        adam_eps=1e-8,
        temp_init=1.75,
        temp_mid=1.2,
        temp_final=0.50,
        # clipping (Tiny has no head group)
        backbone_max_norm=1.5,
        router_max_norm=0.5,
    )
    # Epoch counts / warmups (schedule durations).
    ep = EpochParams(
        train_epochs=100,
        uniform_epochs=10,
        warmup_backbone=15,
        router_warmup=10,
        temp_epochs=25,
    )
    # Data augmentation (mixup / cutmix)
    aug = AugParams(
        use_mixup_cutmix=True,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        cutmix_prob=0.3,
    )
    # ACTIVE aux-loss weights, applied from router_start onward (gated to 0.0
    # before then inside the lit module). label_smoothing=0.10 for the train CE.
    lw = LossWeights(
        z_loss_weight=1e-2,
        spatial_loss_weight=1e-3,
        label_smoothing=0.10,
    )

    tiny_set = get_tinyimagenet_sets(hp.batch_size)
    train_loader = tiny_set['dataloaders']['train']
    val_loader = tiny_set['dataloaders']['val']
    num_classes = tiny_set['num_classes']
    hp.num_classes = num_classes   # set at runtime from the dataset


    print(f'Num classes : {num_classes}')

    print(f'--- Dataset loaded --- \n')

    run_name = f"test {hp.num_experts} experts - Patching-Overlap  | Expert -> Token - Dual Branch Stage EC"

    # Defines checkpointer and Logger
    logger = WandbLogger(
        project="PCE",
        log_model = False,
        name = f'Test-Tiny-{run_name}',
    )
    logger.experiment.define_metric("epoch")
    logger.experiment.define_metric("*", step_metric="epoch")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        filename = 'best-model',
        dirpath = 'checkpoints/',
        save_weights_only = False,
    )
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
        use_static_map=hp.use_static_map,
        unified_router = hp.unified_router,
        uniform_epochs=ep.uniform_epochs,
        task=hp.task,
        )
    # pce = torch.compile(pce, mode="reduce-overhead")

    lit_module = TinyLitModule(
        pce=pce, hparams=hp, loss_weights=lw, aug_params=aug, epoch_params=ep, device=device, static="learnable",
    )
    trainer = pl.Trainer(
        max_epochs=ep.train_epochs,
        logger = logger,
        precision=precision,
        accelerator=device,
        enable_checkpointing= True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=hp.accumulate_grad_batches,
    )

    print(f'--- Start training --- \n')
    if os.path.exists('checkpoints/last.ckpt'):
        trainer.fit(lit_module, train_loader, val_loader, ckpt_path='checkpoints/last.ckpt')
    else:
        trainer.fit(lit_module, train_loader, val_loader)

    logger.experiment.finish()

    del pce, lit_module, trainer, logger
    torch.cuda.empty_cache()
    gc.collect()





