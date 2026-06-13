import pytorch_lightning as pl

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAccuracy

from config import HParams, LossWeights, EpochParams
from train_utils import collect_model_prameters, collect_router_metrics, clip_gradients
from schedulers import backbone_lr_lambda, router_lr_lambda, temp_scheduler, noise_scheduler

from Model.Components.DownsampleResBlock import DownsampleResBlock


class CelebaLitModule(pl.LightningModule):
    """CelebAMask-HQ face parsing (semantic segmentation, 19 classes).

    Modeled on PascalLitModule. Trained FROM SCRATCH: backbone, head and router
    all start at epoch 0 with their own warmups; aux losses active from the start.

    Args:
        static: "learnable" (default) or "not_learnable"; keeps the static map
            (pos_coeff) frozen on unfreeze when "not_learnable".
    """

    def __init__(self, model, hparams: HParams, loss_weights: LossWeights, epoch_params: EpochParams, static: str = "learnable"):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model

        # Controls whether the static map (pos_coeff) is trainable when unfrozen.
        assert static in ("learnable", "not_learnable")
        self.static = static

        # Optimizer / schedule hyperparameters (from HParams)
        self.backbone_lr = hparams.backbone_lr
        self.router_lr = hparams.router_lr
        self.head_lr = hparams.head_lr
        self.weight_decay = hparams.weight_decay
        self.num_classes = hparams.num_classes

        self.adam_betas = hparams.adam_betas
        self.adam_eps = hparams.adam_eps

        self.temp_init = hparams.temp_init
        self.temp_final = hparams.temp_final
        self.temp_epochs = epoch_params.temp_epochs

        self.warmup_backbone = epoch_params.warmup_backbone
        self.head_warmup = epoch_params.head_warmup
        self.router_warmup = epoch_params.router_warmup
        # router_start_epoch == uniform_epochs (derived, not a hyperparameter)
        self.router_start_epoch = epoch_params.uniform_epochs

        self.train_epochs = epoch_params.train_epochs

        # Per-group gradient clip thresholds.
        self.backbone_max_norm = hparams.backbone_max_norm
        self.router_max_norm = hparams.router_max_norm
        self.head_max_norm = hparams.head_max_norm

        self.uniform_epochs = epoch_params.uniform_epochs

        # Loss weights (from LossWeights)
        self.lw = loss_weights
        # Aux losses gated: 0.0 during uniform window, set to LossWeights values
        # from router_start_epoch onward (see on_train_epoch_start).
        self.z_loss_weigth = 0.0
        self.spatial_loss_weight = 0.0
        self.ignore_index = loss_weights.ignore_index
        self.label_smoothing = loss_weights.label_smoothing

        # Training losses (on-device scalars, reduced at epoch end)
        self.train_seg_losses = []
        self.train_aux_losses = []
        self.train_raw_spatial_losses = []
        self.train_weighted_spatial_losses = []
        self.train_raw_z_losses = []
        self.train_weighted_z_losses = []
        self.train_total_losses = []

        # Validation losses
        self.val_seg_losses = []
        self.val_aux_losses = []
        self.val_raw_spatial_losses = []
        self.val_weighted_spatial_losses = []
        self.val_raw_z_losses = []
        self.val_weighted_z_losses = []
        self.val_total_losses = []

        # Snapshot of train router metrics, taken at on_validation_epoch_start
        self._train_router_metrics = {}

        # Per-pixel CE, ignore_index = padding (255).
        self.train_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_index, label_smoothing=self.label_smoothing
        )
        self.val_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Segmentation metrics
        self.metrics = nn.ModuleDict({
            "miou":   JaccardIndex(task="multiclass", num_classes=self.num_classes,
                                   ignore_index=self.ignore_index, average="macro"),
            "pixacc": MulticlassAccuracy(num_classes=self.num_classes,
                                         ignore_index=self.ignore_index, average="micro"),
            "macc":   MulticlassAccuracy(num_classes=self.num_classes,
                                         ignore_index=self.ignore_index, average="macro"),
        })

    def forward(self, x):
        return self.model(x, current_epoch=self.current_epoch, collect_routes=True, collect_debug=False)

    @staticmethod
    def _mean_tensor(values):
        """Mean of a list of detached on-device scalar tensors, materialised once."""
        if not values:
            return 0.0
        return torch.stack(values).float().mean().item()

    def training_step(self, batch, batch_idx):
        data, label = batch
        data, label = data.to(self.device), label.to(self.device)

        # Forward: logits (B, C, H, W)
        logits, spatial_loss, z_loss, _ = self(data)

        seg_loss = self.train_loss(logits, label.long())

        weighted_spatial_loss = spatial_loss * self.spatial_loss_weight
        weighted_z_loss = z_loss * self.z_loss_weigth
        aux_loss = weighted_z_loss + weighted_spatial_loss
        total_loss = seg_loss + aux_loss

        self.train_seg_losses.append(seg_loss.detach())
        self.train_aux_losses.append(aux_loss.detach())
        self.train_raw_spatial_losses.append(spatial_loss.detach())
        self.train_weighted_spatial_losses.append(weighted_spatial_loss.detach())
        self.train_raw_z_losses.append(z_loss.detach())
        self.train_weighted_z_losses.append(weighted_z_loss.detach())
        self.train_total_losses.append(total_loss.detach())

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        data, label = batch
        data, label = data.to(self.device), label.to(self.device)

        logits, spatial_loss, z_loss, _ = self(data)

        seg_loss = self.val_loss(logits, label.long())

        weighted_spatial_loss = spatial_loss * self.spatial_loss_weight
        weighted_z_loss = z_loss * self.z_loss_weigth
        aux_loss = weighted_z_loss + weighted_spatial_loss
        total_loss = seg_loss + aux_loss

        self.val_seg_losses.append(seg_loss.detach())
        self.val_aux_losses.append(aux_loss.detach())
        self.val_raw_spatial_losses.append(spatial_loss.detach())
        self.val_weighted_spatial_losses.append(weighted_spatial_loss.detach())
        self.val_raw_z_losses.append(z_loss.detach())
        self.val_weighted_z_losses.append(weighted_z_loss.detach())
        self.val_total_losses.append(total_loss.detach())

        target = label.long()
        self.metrics["miou"].update(logits, target)
        self.metrics["pixacc"].update(logits, target)
        self.metrics["macc"].update(logits, target)

        return {"loss": total_loss}

    def on_train_epoch_start(self):
        with torch.no_grad():
            self.model.moe_aggregator.reset()

        # Freeze router during uniform window (e < uniform_epochs); from
        # router_start_epoch onward unfreeze, activate aux losses, advance schedules.
        e = self.current_epoch
        if e < self.uniform_epochs:
            self._freeze_router()

        elif e >= self.router_start_epoch:
            self._unfreeze_router()
            self.z_loss_weigth = self.lw.z_loss_weight
            self.spatial_loss_weight = self.lw.spatial_loss_weight

            # Advance temp/noise schedules now that the router trains.
            self.model.router.router_temp = temp_scheduler(
                current_epoch=self.current_epoch,
                router_start_epoch=self.router_start_epoch,
                router_warmup=self.router_warmup,
                train_epochs=self.train_epochs,
                temp_epochs=self.temp_epochs,
                temp_init=self.temp_init,
                temp_final=self.temp_final,
            )
            self.model.router.noise_std = noise_scheduler(
                current_epoch=self.current_epoch,
                router_start_epoch=self.router_start_epoch,
                router_warmup=self.router_warmup,
                train_epochs=self.train_epochs,
                temp_epochs=self.temp_epochs,
            )

    def on_train_epoch_end(self):
        log_dict = {
            'training/seg_loss': self._mean_tensor(self.train_seg_losses),
            'training/aux_loss': self._mean_tensor(self.train_aux_losses),
            'training/raw_spatial_loss': self._mean_tensor(self.train_raw_spatial_losses),
            'training/weighted_spatial_loss': self._mean_tensor(self.train_weighted_spatial_losses),
            'training/raw_z_loss': self._mean_tensor(self.train_raw_z_losses),
            'training/weighted_z_loss': self._mean_tensor(self.train_weighted_z_losses),
            'training/total_loss': self._mean_tensor(self.train_total_losses),
            'LR_backbone': self.optimizer.param_groups[0]['lr'],
            'LR_head': self.optimizer.param_groups[1]['lr'],
            'LR_router': self.optimizer.param_groups[2]['lr'],
            'temp_logits': torch.tensor(self.model.router.router_temp),
        }

        self.train_seg_losses.clear()
        self.train_aux_losses.clear()
        self.train_raw_spatial_losses.clear()
        self.train_weighted_spatial_losses.clear()
        self.train_raw_z_losses.clear()
        self.train_weighted_z_losses.clear()
        self.train_total_losses.clear()

        log_dict.update(self._train_router_metrics)

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self):
        if not self.trainer.sanity_checking:
            self._train_router_metrics = collect_router_metrics('router-train', self.model)

        for m in self.metrics.values():
            m.reset()

        with torch.no_grad():
            self.model.moe_aggregator.reset()

    def on_validation_epoch_end(self):
        log_dict = {
            'validation/seg_loss': self._mean_tensor(self.val_seg_losses),
            'validation/aux_loss': self._mean_tensor(self.val_aux_losses),
            'validation/raw_spatial_loss': self._mean_tensor(self.val_raw_spatial_losses),
            'validation/weighted_spatial_loss': self._mean_tensor(self.val_weighted_spatial_losses),
            'validation/raw_z_loss': self._mean_tensor(self.val_raw_z_losses),
            'validation/weighted_z_loss': self._mean_tensor(self.val_weighted_z_losses),
            'validation/total_loss': self._mean_tensor(self.val_total_losses),
            'validation/mIoU': self.metrics["miou"].compute().item() * 100,
            'validation/pixel_acc': self.metrics["pixacc"].compute().item() * 100,
            'validation/mean_acc': self.metrics["macc"].compute().item() * 100,
        }

        self.val_seg_losses.clear()
        self.val_aux_losses.clear()
        self.val_raw_spatial_losses.clear()
        self.val_weighted_spatial_losses.clear()
        self.val_raw_z_losses.clear()
        self.val_weighted_z_losses.clear()
        self.val_total_losses.clear()

        log_dict.update(collect_router_metrics('router-val', self.model))

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # 3 groups: backbone, head, router (all from scratch on CelebA).
        self.backbone_params, self.router_params, self.head_params = collect_model_prameters(
            self.model, collect_head=True
        )

        # Head follows the backbone schedule but with its own warmup.
        backbone_fn = lambda e: backbone_lr_lambda(e, self.warmup_backbone, self.train_epochs)
        head_fn     = lambda e: backbone_lr_lambda(e, self.head_warmup, self.train_epochs)
        router_fn   = lambda e: router_lr_lambda(e, self.router_start_epoch, self.router_warmup, self.train_epochs)

        # Group order == lr_lambda order passed to LambdaLR.
        self.optimizer = AdamW(
            [
                {'params': self.backbone_params, 'lr': self.backbone_lr, 'weight_decay': self.weight_decay, 'name': 'backbone'},
                {'params': self.head_params,     'lr': self.head_lr,     'weight_decay': self.weight_decay, 'name': 'head'},
                {'params': self.router_params,   'lr': self.router_lr,   'weight_decay': 0.0,               'name': 'router'},
            ], betas=self.adam_betas, eps=self.adam_eps
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[backbone_fn, head_fn, router_fn])

        return [self.optimizer], [self.lr_scheduler]

    def on_after_backward(self):
        # Per-group clipping: the head (from scratch) has its own max_norm.
        clip_gradients(
            backbone_params=self.backbone_params,
            router_params=self.router_params,
            head_params=self.head_params,
            backbone_max_norm=self.backbone_max_norm,
            router_max_norm=self.router_max_norm,
            head_max_norm=self.head_max_norm,
        )

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
