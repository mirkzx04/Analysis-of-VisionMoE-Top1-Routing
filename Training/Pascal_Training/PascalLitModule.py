import pytorch_lightning as pl

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAccuracy

from train_utils import collect_model_prameters, collect_router_metrics, clip_gradients
from schedulers import backbone_lr_lambda, router_lr_lambda, temp_scheduler, noise_scheduler

class PascalLitModule(pl.LightningModule):
    def __init__(
        self,
        model,
        backbone_lr,
        router_lr,
        head_lr,
        weight_decay,
        num_classes,
        train_epochs,
        temp_init,
        temp_final,
        temp_epochs
    ):

        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.backbone_lr = backbone_lr
        self.router_lr = router_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        self.temp_init = temp_init
        self.temp_final = temp_final
        self.temp_epochs = temp_epochs

        # Pretrained backbone: short warmup. Head and router start from scratch.
        # No uniform/frozen phase on Pascal: the backbone is already trained, so
        # the router is optimized from epoch 0. router_start_epoch=0 anchors the
        # router LR and the temp/noise schedules at the start (no disabled window).
        self.warmup_backbone = 5
        self.head_warmup = 10
        self.router_warmup = 5
        self.router_start_epoch = 0

        self.train_epochs = train_epochs

        # Router trained from the start -> aux losses active from epoch 0.
        self.z_loss_weigth = 1e-2
        self.spatial_loss_weight = 1e-3

        # Training losses (on-device scalars, reduced once at the end of the epoch)
        self.train_seg_losses = []
        self.train_aux_losses = []
        self.train_raw_spatial_losses = []
        self.train_weighted_spatial_losses = []
        self.train_total_losses = []

        # Validation losses
        self.val_seg_losses = []
        self.val_aux_losses = []
        self.val_raw_spatial_losses = []
        self.val_weighted_spatial_losses = []
        self.val_total_losses = []

        # Snapshot of train router metrics, taken at on_validation_epoch_start
        # (after train batches, before the aggregator is reset for validation).
        self._train_router_metrics = {}

        # Loss function: per-pixel CE, 255 = ignored VOC void/border.
        self.train_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.val_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

        # Segmentation metrics. In a ModuleDict, Lightning moves them to the
        # module device (no manual .to(device)). They accumulate over the entire
        # validation set via an internal confusion matrix: update() per batch,
        # compute() at the end of the epoch.
        self.metrics = nn.ModuleDict({
            "miou":   JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255, average="macro"),
            "pixacc": MulticlassAccuracy(num_classes=num_classes, ignore_index=255, average="micro"),
            "macc":   MulticlassAccuracy(num_classes=num_classes, ignore_index=255, average="macro"),
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
        aux_loss = (z_loss * self.z_loss_weigth) + weighted_spatial_loss
        total_loss = seg_loss + aux_loss

        self.train_seg_losses.append(seg_loss.detach())
        self.train_aux_losses.append(aux_loss.detach())
        self.train_raw_spatial_losses.append(spatial_loss.detach())
        self.train_weighted_spatial_losses.append(weighted_spatial_loss.detach())
        self.train_total_losses.append(total_loss.detach())

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        data, label = batch
        data, label = data.to(self.device), label.to(self.device)

        logits, spatial_loss, z_loss, _ = self(data)

        seg_loss = self.val_loss(logits, label.long())

        weighted_spatial_loss = spatial_loss * self.spatial_loss_weight
        aux_loss = (z_loss * self.z_loss_weigth) + weighted_spatial_loss
        total_loss = seg_loss + aux_loss

        self.val_seg_losses.append(seg_loss.detach())
        self.val_aux_losses.append(aux_loss.detach())
        self.val_raw_spatial_losses.append(spatial_loss.detach())
        self.val_weighted_spatial_losses.append(weighted_spatial_loss.detach())
        self.val_total_losses.append(total_loss.detach())

        target = label.long()
        self.metrics["miou"].update(logits, target)
        self.metrics["pixacc"].update(logits, target)
        self.metrics["macc"].update(logits, target)

        return {"loss": total_loss}

    def on_train_epoch_start(self):
        with torch.no_grad():
            self.model.moe_aggregator.reset()

        # Router trained from the start: just advance the temp/noise schedules.
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
        self.train_total_losses.clear()

        log_dict.update(self._train_router_metrics)

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self):
        # This hook runs after the training batches but before any validation
        # accumulation, so the aggregator still holds TRAIN routing data here.
        # Snapshot it before resetting the aggregator for validation. Skip during
        # Lightning's pre-training sanity check (no training has happened yet).
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
            'validation/total_loss': self._mean_tensor(self.val_total_losses),
            'validation/mIoU': self.metrics["miou"].compute().item() * 100,
            'validation/pixel_acc': self.metrics["pixacc"].compute().item() * 100,
            'validation/mean_acc': self.metrics["macc"].compute().item() * 100,
        }

        self.val_seg_losses.clear()
        self.val_aux_losses.clear()
        self.val_raw_spatial_losses.clear()
        self.val_weighted_spatial_losses.clear()
        self.val_total_losses.clear()

        log_dict.update(collect_router_metrics('router-val', self.model))

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # 3 groups: backbone (pretrained), head (from scratch), router (from scratch).
        self.backbone_params, self.router_params, self.head_params = collect_model_prameters(
            self.model, collect_head=True
        )

        # The head follows the backbone schedule, but with its own warmup (head_warmup).
        backbone_fn = lambda e: backbone_lr_lambda(e, self.warmup_backbone, self.train_epochs)
        head_fn     = lambda e: backbone_lr_lambda(e, self.head_warmup, self.train_epochs)
        router_fn   = lambda e: router_lr_lambda(e, self.router_start_epoch, self.router_warmup, self.train_epochs)

        # Group order == lr_lambda order passed to LambdaLR.
        self.optimizer = AdamW(
            [
                {'params': self.backbone_params, 'lr': self.backbone_lr, 'weight_decay': self.weight_decay, 'name': 'backbone'},
                {'params': self.head_params,     'lr': self.head_lr,     'weight_decay': self.weight_decay, 'name': 'head'},
                {'params': self.router_params,   'lr': self.router_lr,   'weight_decay': 0.0,               'name': 'router'},
            ], betas=(0.9, 0.98), eps=1e-8
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[backbone_fn, head_fn, router_fn])

        return [self.optimizer], [self.lr_scheduler]

    def on_after_backward(self):
        # Per-group clipping: the head (from scratch) has its own max_norm.
        clip_gradients(
            backbone_params=self.backbone_params,
            router_params=self.router_params,
            head_params=self.head_params,
            collect_head=True,
            backbone_max_norm=1.5,
            router_max_norm=0.5,
            head_max_norm=1.0,
        )
