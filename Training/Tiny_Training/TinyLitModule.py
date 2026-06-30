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
from train_utils import collect_model_prameters, collect_router_metrics, clip_gradients

from config import HParams, LossWeights, AugParams, EpochParams, RouterTypesParams

class TinyLitModule(pl.LightningModule):
    def __init__(
                self,
                pce,
                hparams: HParams,
                loss_weights: LossWeights,
                aug_params: AugParams,
                epoch_params: EpochParams,
                device,
                static: str = "learnable",
                router_types: RouterTypesParams = None,
            ):

        """
        Initialize the TinyLitModule.

        Args:
            pce: PCENetwork model.
            hparams (HParams): model / optimizer / schedule config.
            loss_weights (LossWeights): aux-loss weights + CE config; ACTIVE values
                applied from router_start onward (gated to 0.0 before).
            aug_params (AugParams): data-augmentation config (mixup / cutmix).
            epoch_params (EpochParams): epoch counts and warmups (schedule durations).
            device (str): Device to use (for the accuracy metrics).
            static (str): "learnable" (default) or "not_learnable"; keeps the static
                map (pos_coeff) frozen on unfreeze when "not_learnable".
        Returns:
            None
        """

        super().__init__()
        # Model and optimizer parameters
        self.model = pce
        self.hp = hparams
        self.lw = loss_weights

        # Controls whether the static map (pos_coeff) is trainable when unfrozen.
        assert static in ("learnable", "not_learnable")
        self.static = static
        self.router_types = router_types

        self.lr = hparams.backbone_lr
        self.router_lr = hparams.router_lr
        self.weight_decay = hparams.weight_decay

        self.temp_init = hparams.temp_init # Logits router temperature
        self.temp_mid = hparams.temp_mid
        self.temp_final = hparams.temp_final
        self.temp_epochs = epoch_params.temp_epochs
        self.num_classes = hparams.num_classes

        self.router_mul = 2.0
        self.warmup_backbone = epoch_params.warmup_backbone
        # router_start_epoch == uniform_epochs (derived, not a hyperparameter)
        self.router_start_epoch = epoch_params.uniform_epochs
        self.router_warmup = epoch_params.router_warmup
        self.use_augmentation = True

        self.train_epochs = epoch_params.train_epochs
        self.uniform_epochs = epoch_params.uniform_epochs

        # Per-group gradient-clip thresholds (Tiny has no head group).
        self.backbone_max_norm = hparams.backbone_max_norm
        self.router_max_norm = hparams.router_max_norm

        # Training losses
        self.train_class_losses = []
        self.train_aux_losses = []
        self.train_raw_z_losses = []
        self.train_weighted_z_losses = []
        self.train_total_losses = []

        # Validation losses
        self.val_class_losses = []
        self.val_aux_losses = []
        self.val_raw_z_losses = []
        self.val_weighted_z_losses = []
        self.val_total_losses = []
        
        self.gradient_norm_router = []
        self.gradient_norm_backbone = []
        self.grad_metrics_interval = 25
        self._grad_metrics_steps = 0
        self.router_detail_metrics_interval = 25

        # Best validation loss
        self.best_val_loss = float('+inf')

        self.use_mixup_cutmix = aug_params.use_mixup_cutmix
        self.mixup_alpha      = aug_params.mixup_alpha
        self.cutmix_alpha     = aug_params.cutmix_alpha
        self.cutmix_prob      = aug_params.cutmix_prob

        # Aux losses gated off until router_start (set to active values in on_train_epoch_start)
        self.z_loss_weigth = 0.0

        # Accuracy metrics
        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1).to(device),
            'top5_train' : Accuracy(task='multiclass', num_classes=self.num_classes, top_k=5).to(device),

            'top1_val' : Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1).to(device),
            'top5_val' : Accuracy(task='multiclass', num_classes=self.num_classes, top_k=5).to(device)
        }
        # Loss function
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.train_loss = torch.nn.CrossEntropyLoss(label_smoothing=loss_weights.label_smoothing) # M

    def setup(self, stage=None):
        # Log the router-type selection (provenance) to the experiment config once the
        # logger is attached. Flat primitive fields only, so W&B serialization is safe.
        if self.router_types is not None and self.logger is not None:
            try:
                self.logger.log_hyperparams(
                    {f"router_types/{k}": v for k, v in vars(self.router_types).items()}
                )
            except Exception:
                pass

    def forward(self, x, force_specialized = False, class_labels = None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits from the model.
        """
        return self.model(
            x,
            current_epoch=self.current_epoch,
            collect_routes=True,
            collect_debug=False,
            class_labels=class_labels,
        )

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

            logits, z_loss, _div, _  = self(data)

            # Loss = lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b)
            class_loss = (
                lam * self.train_loss(logits, targets_a)
                + (1.0 - lam) * self.train_loss(logits, targets_b)
            )

        else:
            logits, z_loss, _div, _ = self(data, class_labels=labels)
            class_loss = self.train_loss(logits, labels)

        weighted_z_loss = z_loss * self.z_loss_weigth
        aux_loss = weighted_z_loss
        total_loss = class_loss + aux_loss

        # Detached on-device scalars, reduced once per epoch (avoids per-step GPU->CPU sync)
        self.train_class_losses.append(class_loss.detach())
        self.train_aux_losses.append(aux_loss.detach())
        self.train_raw_z_losses.append(z_loss.detach())
        self.train_weighted_z_losses.append(weighted_z_loss.detach())

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
            post_block_params=self.post_block_params,
            backbone_max_norm=self.backbone_max_norm,
            router_max_norm=self.router_max_norm,
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
            self.z_loss_weigth = self.lw.z_loss_weight

            self.model.router.router_temp = temp_scheduler(
                current_epoch=self.current_epoch, 
                router_start_epoch=self.router_start_epoch, 
                router_warmup=self.router_warmup, 
                train_epochs=self.train_epochs,
                temp_epochs=self.temp_epochs, 
                temp_init=self.temp_init, 
                temp_final=self.temp_final
            )
            self.model.router.noise_std = noise_scheduler(
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
            'training/raw_z_loss' : self._mean_tensor(self.train_raw_z_losses),
            'training/weighted_z_loss' : self._mean_tensor(self.train_weighted_z_losses),
            'training/train_total_loss' : self._mean_tensor(self.train_total_losses),

            'training/train_top1' : self.accuracy_metrics['top1_train'].compute().item() * 100,
            'training/train_top5' : self.accuracy_metrics['top5_train'].compute().item() * 100,
            'LR_backbone : ' : self.optimizer.param_groups[0]['lr'],
            'LR_Router' : self.optimizer.param_groups[1]['lr'],
            'Gradient norm backbone' : self._mean_float(self.gradient_norm_backbone),
            'Gradient norm router' : self._mean_float(self.gradient_norm_router),
            'temp_logits' : torch.tensor(self.model.router.router_temp),
        }

        self.train_class_losses.clear()
        self.train_aux_losses.clear()
        self.train_raw_z_losses.clear()
        self.train_weighted_z_losses.clear()
        self.train_total_losses.clear()
        self.train_class_losses.clear()
        self.train_aux_losses.clear()
        
        self.gradient_norm_router.clear()
        self.gradient_norm_backbone.clear()

        log_dict.update(collect_router_metrics('router-train', self.model))

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

        logits, z_loss, _div, _ = self(data, class_labels=labels)

        class_loss = self.val_loss(logits, labels)

        weighted_z_loss = z_loss * self.z_loss_weigth
        aux_loss = weighted_z_loss
        total_loss = class_loss + aux_loss

        # Detached on-device scalars, reduced once per epoch (avoids per-step GPU->CPU sync)
        self.val_class_losses.append(class_loss.detach())
        self.val_aux_losses.append(aux_loss.detach())
        self.val_raw_z_losses.append(z_loss.detach())
        self.val_weighted_z_losses.append(weighted_z_loss.detach())
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
            'validation/raw_z_loss' : self._mean_tensor(self.val_raw_z_losses),
            'validation/weighted_z_loss' : self._mean_tensor(self.val_weighted_z_losses),
            'validation/val_total_loss' : self._mean_tensor(self.val_total_losses),
            'validation/val_top1' : self.accuracy_metrics['top1_val'].compute().item() * 100,
            'validation/val_top5' : self.accuracy_metrics['top5_val'].compute().item() * 100,
        }

        self.val_class_losses.clear()
        self.val_aux_losses.clear()
        self.val_raw_z_losses.clear()
        self.val_weighted_z_losses.clear()
        self.val_total_losses.clear()

        log_dict.update(collect_router_metrics('router-val', self.model))

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):

        self.backbone_params, self.router_params, self.post_block_params = collect_model_prameters(
            self.model, collect_post_block=True
        )

        # LambdaLR vuole callable(epoch); chiudiamo le fn di schedulers.py sui soli `epoch`
        backbone_fn = lambda e: backbone_lr_lambda(e, self.warmup_backbone, self.train_epochs)
        router_fn = lambda e: router_lr_lambda(e, self.router_start_epoch, self.router_warmup, self.train_epochs)
        # post_block segue lo schedule del backbone; la sua lr base è già 0.1 * lr backbone.
        post_block_fn = lambda e: backbone_lr_lambda(e, self.warmup_backbone, self.train_epochs)

        self.optimizer = AdamW(
            [
                {'params': self.backbone_params, 'lr': self.lr, 'weight_decay': self.weight_decay, 'name': 'backbone'},
                {'params': self.router_params, 'lr': self.router_lr, 'weight_decay': 0.0, 'name': 'router'},
                {'params': self.post_block_params, 'lr': 0.1 * self.lr, 'weight_decay': self.weight_decay, 'name': 'post_block'},
            ], betas=self.hp.adam_betas, eps=self.hp.adam_eps
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[backbone_fn, router_fn, post_block_fn])

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
                # Keep a frozen static map frozen when static="not_learnable".
                if getattr(l.router_gate, "is_static_map", False) and self.static == "not_learnable":
                    continue
                for p in l.router_gate.parameters():
                    p.requires_grad_(True)

        for p in self.model.router.parameters():
            p.requires_grad_(True)

    def _freeze_post_block(self):
        for n, p in self.model.named_parameters():
            if 'post_block' in n:
                p.requires_grad_(False)

    def _unfreeze_post_block(self):
        for n, p in self.model.named_parameters():
            if 'post_block' in n:
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
