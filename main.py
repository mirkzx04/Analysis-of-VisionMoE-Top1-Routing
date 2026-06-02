import os
import numpy as np 
import matplotlib.pyplot as plt
import json

import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
gc.collect()
torch.cuda.empty_cache()

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from Datasets_Classes.Cifar10 import CIFAR10Dataset, CIFAR10TrainDataset, CIFAR10ValidationDataset
from Datasets_Classes.Cifar100 import CIFAR100Dataset, CIFAR100TrainDataset, CIFAR100ValidationDataset
from Datasets_Classes.TinyImageNet import TinyImageNetDataset, TinyImageNetTrainDataset, TinyImageNetValidationDataset

from Model.PCE import PCENetwork

# from Datasets_Classes.PascalVOC import PascalVOCDataset, PascalVOCTrainDataset, PascalVOCValidationDataset
# from Datasets_Classes.PatchExtractor import PatchExtractor

from EMADiffLitModule import EMADiffLitModule

def count_number_of_classes(train_labels, val_labels):   
    return len(np.unique(np.concatenate([train_labels, val_labels])).tolist())

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'-- Start with device : {device} ---')
    print('\n ------------------------ \n')

    # Hyperparameters of model
    num_exp = 16
    layer_number = 8
    patch_size = 16
    lr = 0.001
    router_lr = 1e-4
    weight_decay = 1e-3 # M

    # Hyperparameters of router
    capacity_factor_train = 1.00
    capacity_factor_val = 1.00
    halo_for_patches = 2

    temp_init = 1.75
    temp_mid = 1.2
    temp_final = 0.75
    temp_epochs = 25
    
    # Training metrics
    train_epochs = 80
    uniform_epochs = 10
    batch_size = 64

    tiny_set = get_tinyimagenet_sets(batch_size)
    train_loader = tiny_set['dataloaders']['train']
    val_loader = tiny_set['dataloaders']['val']
    num_classes = tiny_set['num_classes']


    print(f'Num classes : {num_classes}')

    print(f'--- Dataset loaded --- \n')

    run_name = f"test {num_exp} experts - Patching-Overlap  | Expert -> Token | Cov Losses new W"

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
        num_experts = num_exp,
        layer_number = layer_number,
        patch_size = patch_size,
        num_classes=num_classes,
        router_temp=temp_init,
        capacity_factor_train = capacity_factor_train,
        capacity_factor_val = capacity_factor_val,
        halo_for_patches=halo_for_patches,
        use_static_map=False
        )
    # pce = torch.compile(pce, mode="reduce-overhead")

    lit_module = EMADiffLitModule(
        pce=pce, lr=lr, weight_decay=weight_decay, device=device, train_epochs=train_epochs, 
        uniform_epochs=uniform_epochs, temp_init=temp_init, temp_mid = temp_mid, 
        temp_final=temp_final,temp_epochs=temp_epochs, num_classes=num_classes, router_lr = router_lr,
    )
    trainer = pl.Trainer(
        max_epochs=train_epochs,
        logger = logger,
        precision='32',
        accelerator=device,
        enable_checkpointing= True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=2,
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





    