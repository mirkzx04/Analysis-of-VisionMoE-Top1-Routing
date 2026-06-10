from json import load
from tkinter import BaseWidget
import pytorch_lightning as pl
import torch
from torch.mps import current_allocated_memory
from torch.nn.utils.spectral_norm import SpectralNormLoadStateDictPreHook
import wandb as wb
import math
import random

from torch.nn import functional as F
from torch.optim import AdamW, Adam
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from torchmetrics import Accuracy

import torch.nn as nn

from Model.Components.DownsampleResBlock import DownsampleResBlock
from schedulers import temp_scheduler, noise_scheduler, backbone_lr_lambda, router_lr_lambda
from utils import collect_model_prameters, collect_router_metrics, clip_gradients

class TinyLitModule(pl.LightningModule):
    def __init__(
                self,
                pce,
                lr,
                router_lr,
                weight_decay,
                num_classes,
                device,
                train_epochs,
                uniform_epochs,
                temp_init,
                temp_mid,
                temp_final,
                temp_epochs,
            ):

        """
        Initialize the EMADiffLitModule.

        Args:
            num_experts (int): Number of experts.
            layer_number (int): Number of layers.
            patch_size (int): Patch size.
            dropout (float): Dropout rate.
            num_classes (int): Number of classes.
            nucleus_sampling_p (float): Nucleus sampling probability.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.
            augmentation: Data augmentation object.
            class_names (list): List of class names.
            device (str): Device to use.
        Returns:
            None
        """

        super().__init__()
        # Model and optimizer parameters
        self.model = pce
        self.lr = lr
        self.router_lr = router_lr
        self.weight_decay = weight_decay

        self.temp_init = temp_init # Logits router temperature
        self.temp_mid = temp_mid
        self.temp_final = temp_final
        self.temp_epochs = temp_epochs
        self.num_classes = num_classes

        self.router_mul = 2.0
        self.warmup_backbone = 15
        self.router_start_epoch = uniform_epochs
        self.router_warmup = 10
        self.use_augmentation = True

        self.train_epochs = train_epochs
        self.uniform_epochs = uniform_epochs

        # Training losses
        self.train_class_losses = []
        self.train_aux_losses = []
        self.train_raw_spatial_losses = []
        self.train_weighted_spatial_losses = []
        self.train_total_losses = []

        # Validation losses
        self.val_class_losses = []
        self.val_aux_losses = []
        self.val_raw_spatial_losses = []
        self.val_weighted_spatial_losses = []
        self.val_total_losses = []
        
        self.gradient_norm_router = []
        self.gradient_norm_backbone = []
        self.grad_metrics_interval = 25
        self._grad_metrics_steps = 0
        self.router_detail_metrics_interval = 25

        # Best validation loss
        self.best_val_loss = float('+inf')

        self.use_mixup_cutmix = True
        self.mixup_alpha      = 0.2
        self.cutmix_alpha     = 1.0
        self.cutmix_prob      = 0.3

        self.z_loss_weigth = 0.0
        self.spatial_loss_weight = 0.0

        # Accuracy metrics
        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'top5_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device),

            'top1_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'top5_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
        }
        # Loss function
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.train_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.10) # M


    def forward(self, x, force_specialized = False):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits from the model.
        """
        return self.model(x, current_epoch=self.current_epoch, collect_routes=True, collect_debug=False)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): Batch of data (inputs, labels).
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing predictions, losses, and batch index.
        """
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        if self.use_mixup_cutmix:
            r = random.random()
            if r < self.cutmix_prob:
                # CutMix
                data, targets_a, targets_b, lam = self._cutmix_batch(data, labels)
            else:
                # Mixup
                data, targets_a, targets_b, lam = self._mixup_batch(data, labels)

            logits, spatial_loss, z_loss, _  = self(data)

            # Loss = lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b)
            class_loss = (
                lam * self.train_loss(logits, targets_a)
                + (1.0 - lam) * self.train_loss(logits, targets_b)
            )

        else:
            logits, spatial_loss, z_loss, _ = self(data)
            class_loss = self.train_loss(logits, labels)

        weighted_spatial_loss = spatial_loss * self.spatial_loss_weight
        aux_loss = (z_loss * self.z_loss_weigth) + weighted_spatial_loss
        total_loss = class_loss + aux_loss

        # Store detached on-device scalars and reduce once per epoch (see
        # on_train_epoch_end) instead of forcing a GPU->CPU sync every step.
        self.train_class_losses.append(class_loss.detach())
        self.train_aux_losses.append(aux_loss.detach())
        self.train_raw_spatial_losses.append(spatial_loss.detach())
        self.train_weighted_spatial_losses.append(weighted_spatial_loss.detach())

        self.train_total_losses.append(total_loss.detach())

        if self.num_classes >= 5:
            if self.use_augmentation and labels.dim() > 1 :
                labels_accuracy = torch.argmax(labels, dim = 1)
            else :
                labels_accuracy = labels

            self.accuracy_metrics['top1_train'].update(logits, labels_accuracy)
            self.accuracy_metrics['top5_train'].update(logits, labels_accuracy)

        return {'loss' : total_loss}

    def on_after_backward(self):
        """
        Calculate the gradient norm after backward pass.

        Returns:
            None
        """
        clip_gradients(
            backbone_params=self.backbone_params,
            router_params=self.router_params,
            backbone_max_norm=1.5,
            router_max_norm=0.5,
        )

    def on_train_epoch_start(self):
        self.accuracy_metrics['top1_train'].reset()
        self.accuracy_metrics['top5_train'].reset()

        with torch.no_grad():
            self.model.moe_aggregator.reset()

        e = self.current_epoch
        if e < self.uniform_epochs:
            self._freeze_router()

        elif e >= self.router_start_epoch:
            self._unfreeze_router()
            self.z_loss_weigth = 1e-2
            self.spatial_loss_weight = 1e-3

            self.model.router.router_temp = temp_scheduler(
                current_epoch=self.current_epoch, 
                router_start_epoch=self.router_start_epoch, 
                router_warmup=self.router_warmup, 
                train_epochs=self.train_epochs,
                temp_epochs=self.temp_epochs, 
                temp_init=self.temp_init, 
                temp_final=self.temp_final
            )
            self.model.router.noise_std = self.noise_scheduler(
                current_epoch=self.current_epoch, 
                router_start_epoch = self.router_start_epoch,
                router_warmup = self.router_warmup,
                train_epochs = self.train_epochs, 
                temp_epochs = self.temp_epochs 
            )

    @staticmethod
    def _mean_float(values):
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _mean_tensor(values):
        """Mean of a list of detached on-device scalar tensors, materialised once."""
        if not values:
            return 0.0
        return torch.stack(values).float().mean().item()

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch to compute and reset average losses.

        Returns:
            None
        """

        # Log dictionary
        log_dict = {
            'training/train_class_loss' : self._mean_tensor(self.train_class_losses),
            'training/train_aux_loss' : self._mean_tensor(self.train_aux_losses),
            'training/raw_spatial_loss' : self._mean_tensor(self.train_raw_spatial_losses),
            'training/weighted_spatial_loss' : self._mean_tensor(self.train_weighted_spatial_losses),
            'training/train_total_loss' : self._mean_tensor(self.train_total_losses),

            'training/train_top1' : self.accuracy_metrics['top1_train'].compute().item() * 100,
            'training/train_top5' : self.accuracy_metrics['top5_train'].compute().item() * 100,
            'LR_backbone : ' : self.optimizer.param_groups[0]['lr'],
            'LR_Router' : self.optimizer.param_groups[2]['lr'],
            'Gradient norm backbone' : self._mean_float(self.gradient_norm_backbone),
            'Gradient norm router' : self._mean_float(self.gradient_norm_router),
            'temp_logits' : torch.tensor(self.model.router.router_temp),
        }

        self.train_class_losses.clear()
        self.train_aux_losses.clear()
        self.train_raw_spatial_losses.clear()
        self.train_weighted_spatial_losses.clear()
        self.train_total_losses.clear()
        self.train_class_losses.clear()
        self.train_aux_losses.clear()
        
        self.gradient_norm_router.clear()
        self.gradient_norm_backbone.clear()

        log_dict.update(collect_router_metrics('router-train'))

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): Batch of data (inputs, labels).
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing predictions, losses, batch index, and loss histories.
        """
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        logits, spatial_loss, z_loss, _ = self(data)

        class_loss = self.val_loss(logits, labels)

        weighted_spatial_loss = spatial_loss * self.spatial_loss_weight
        aux_loss = ((z_loss * self.z_loss_weigth) + weighted_spatial_loss)
        total_loss = class_loss + aux_loss

        # Detached on-device scalars, reduced once per epoch (see
        # on_validation_epoch_end) to avoid a GPU->CPU sync every step.
        self.val_class_losses.append(class_loss.detach())
        self.val_aux_losses.append(aux_loss.detach())
        self.val_raw_spatial_losses.append(spatial_loss.detach())
        self.val_weighted_spatial_losses.append(weighted_spatial_loss.detach())
        self.val_total_losses.append(total_loss.detach())

        if self.num_classes >= 5:
            self.accuracy_metrics['top1_val'].update(logits, labels)
            self.accuracy_metrics['top5_val'].update(logits, labels)

        return {'loss' : total_loss}

    def on_validation_epoch_start(self):
        self.accuracy_metrics['top1_val'].reset()
        self.accuracy_metrics['top5_val'].reset()

        with torch.no_grad():
            self.model.moe_aggregator.reset()

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to compute and reset average losses.

        Returns:
            None
        """
        # Log dictionary
        log_dict = {
            'validation/val_class_loss' : self._mean_tensor(self.val_class_losses),
            'validation/val_aux_loss' : self._mean_tensor(self.val_aux_losses),
            'validation/raw_spatial_loss' : self._mean_tensor(self.val_raw_spatial_losses),
            'validation/weighted_spatial_loss' : self._mean_tensor(self.val_weighted_spatial_losses),
            'validation/val_total_loss' : self._mean_tensor(self.val_total_losses),
            'validation/val_top1' : self.accuracy_metrics['top1_val'].compute().item() * 100,
            'validation/val_top5' : self.accuracy_metrics['top5_val'].compute().item() * 100,
        }

        self.val_class_losses.clear()
        self.val_aux_losses.clear()
        self.val_raw_spatial_losses.clear()
        self.val_weighted_spatial_losses.clear()
        self.val_total_losses.clear()

        log_dict.update(collect_router_metrics('router-val'))

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):

        self.backbone_params, self.router_params = collect_model_prameters(self.model)

        base_lr = self.lr
        router_lr = self.router_lr
        wd = self.weight_decay

        backbone_lr_scheduler = backbone_lr_lambda(epoch=self.current_epoch, warmup_backbone=self.warmup_backbone)
        router_lr_scheduler = router_lr_lambda(epoch=self.current_epoch, router_start_epoch=self.router_start_epoch, warmup_router=self.router_warmup)
    
        # Optimizer 2e-5
        self.optimizer = AdamW(
            [
                {'params': backbone_params, 'lr': base_lr, 'weight_decay': wd, 'name' : 'backbone'}, # Conv and Linear
                {'params' : backbone_norm, 'lr': base_lr, 'weight_decay': 0, 'name' : 'backbone_norm'},
                {'params': router_params, 'lr': router_lr, 'weight_decay': 0, 'name' : 'router_w'},
                {'params': router_pos_params, 'lr': router_lr * 4,  'weight_decay': 0,  'name': 'router_position'},
            ], betas=(0.9, 0.98), eps=1e-8
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[
            backbone_lr_scheduler,
            backbone_lr_scheduler,
            router_lr_scheduler,
            router_lr_scheduler,
        ])

        return [self.optimizer], [self.lr_scheduler]

    def _freeze_router(self):
        for l in self.model.layers:
            if not isinstance(l, DownsampleResBlock):
                for p in l.router_gate.parameters():
                    p.requires_grad_(False)

        for p in self.model.router.parameters():
            p.requires_grad_(False)

    def _unfreeze_router(self):
        for l in self.model.layers:
            if not isinstance(l, DownsampleResBlock):
                for p in l.router_gate.parameters():
                    p.requires_grad_(True)

        for p in self.model.router.parameters():
            p.requires_grad_(True)

    # MIXUP - CutMix utils
    def _sample_lambda(self, alpha) :
        """
        Sample lambda from beta
        """
        if alpha <= 0:
            return 1.0

        beta_dist = torch.distributions.Beta(alpha, alpha)
        lam = beta_dist.sample().item()
        return float(lam)

    def _mixup_batch(self, x, y):
        """
        applly Mixup on batch
        """
        if self.mixup_alpha <= 0:
            return x,y,y,1.0

        lam = self._sample_lambda(self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1.0 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _rand_box(self, size, lam):
        """
        Compute a random box for CutMix
        size : (B, C, H, W)
        """

        B, C, H, W = size
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Center of box
        cx = torch.randint(W, (1,), device=self.device).item()
        cy = torch.randint(H, (1,), device=self.device).item()

        bbx1 = max(cx - cut_w // 2, 0)
        bby1 = max(cy - cut_h // 2, 0)
        bbx2 = min(cx + cut_w // 2, W)
        bby2 = min(cy + cut_h // 2, H)

        return bbx1, bby1, bbx2, bby2

    def _cutmix_batch(self, x, y):
        """
        Apply cutmix on batch
        """
        if self.cutmix_alpha <= 0.0:
            return x, y, y, 1.0

        lam = self._sample_lambda(self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self._rand_box(x.size(), lam)

        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        cut_area = (bbx2 - bbx1) * (bby2 - bby1)
        lam = 1.0 - cut_area / float(x.size(2) * x.size(3))

        return x, y_a, y_b, lam
