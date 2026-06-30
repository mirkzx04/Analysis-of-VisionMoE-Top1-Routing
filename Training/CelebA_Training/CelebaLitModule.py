import pytorch_lightning as pl

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAccuracy

from config import HParams, LossWeights, EpochParams, RouterTypesParams
from train_utils import collect_model_prameters, collect_router_metrics, clip_gradients, compute_patch_class
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

    def __init__(self, model, hparams: HParams, loss_weights: LossWeights, epoch_params: EpochParams, static: str = "learnable", router_types: RouterTypesParams = None):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'router_types'])

        self.model = model

        # Controls whether the static map (pos_coeff) is trainable when unfrozen.
        assert static in ("learnable", "not_learnable")
        self.static = static
        self.router_types = router_types

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
        self.div_loss_weight = 0.0
        self.ignore_index = loss_weights.ignore_index
        self.label_smoothing = loss_weights.label_smoothing

        # Training losses (on-device scalars, reduced at epoch end)
        self.train_seg_losses = []
        self.train_aux_losses = []
        self.train_raw_z_losses = []
        self.train_weighted_z_losses = []
        self.train_raw_div_losses = []
        self.train_weighted_div_losses = []
        self.train_total_losses = []

        # Validation losses
        self.val_seg_losses = []
        self.val_aux_losses = []
        self.val_raw_z_losses = []
        self.val_weighted_z_losses = []
        self.val_raw_div_losses = []
        self.val_weighted_div_losses = []
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

    def forward(self, x, patch_class=None):
        return self.model(
            x,
            current_epoch=self.current_epoch,
            collect_routes=True,
            collect_debug=False,
            patch_class=patch_class,
        )

    def _patch_class(self, label):
        """Per-patch dominant class [B, P] from the GT mask, for MI(exp, class).

        Uses the model's fixed patch grid (7x7 -> P=49) and the CE ignore_index
        (255 void) so the definition matches Testing/experiments/mi_pos_class.py.
        """
        grid_h, grid_w = self.model.patch_grid
        return compute_patch_class(
            label.long(), grid_h, grid_w, self.num_classes, self.ignore_index
        )

    @staticmethod
    def _mean_tensor(values):
        """Mean of a list of detached on-device scalar tensors, materialised once."""
        if not values:
            return 0.0
        return torch.stack(values).float().mean().item()

    def training_step(self, batch, batch_idx):
        data, label = batch
        data, label = data.to(self.device), label.to(self.device)

        # Forward: logits (B, C, H, W). patch_class drives the MI(exp, class) metric.
        logits, z_loss, div_loss, _ = self(data, patch_class=self._patch_class(label))

        seg_loss = self.train_loss(logits, label.long())

        weighted_z_loss = z_loss * self.z_loss_weigth
        weighted_div_loss = div_loss * self.div_loss_weight
        aux_loss = weighted_z_loss + weighted_div_loss
        total_loss = seg_loss + aux_loss

        self.train_seg_losses.append(seg_loss.detach())
        self.train_aux_losses.append(aux_loss.detach())
        self.train_raw_z_losses.append(z_loss.detach())
        self.train_weighted_z_losses.append(weighted_z_loss.detach())
        self.train_raw_div_losses.append(div_loss.detach())
        self.train_weighted_div_losses.append(weighted_div_loss.detach())
        self.train_total_losses.append(total_loss.detach())

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        data, label = batch
        data, label = data.to(self.device), label.to(self.device)

        logits, z_loss, div_loss, _ = self(data, patch_class=self._patch_class(label))

        seg_loss = self.val_loss(logits, label.long())

        weighted_z_loss = z_loss * self.z_loss_weigth
        weighted_div_loss = div_loss * self.div_loss_weight
        aux_loss = weighted_z_loss + weighted_div_loss
        total_loss = seg_loss + aux_loss

        self.val_seg_losses.append(seg_loss.detach())
        self.val_aux_losses.append(aux_loss.detach())
        self.val_raw_z_losses.append(z_loss.detach())
        self.val_weighted_z_losses.append(weighted_z_loss.detach())
        self.val_raw_div_losses.append(div_loss.detach())
        self.val_weighted_div_losses.append(weighted_div_loss.detach())
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
            self._freeze_post_block()

        elif e >= self.router_start_epoch:
            self._unfreeze_router()
            self._unfreeze_post_block()
            
            self.z_loss_weigth = self.lw.z_loss_weight
            self.div_loss_weight = self.lw.diversity_loss_weight 

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
            'training/raw_z_loss': self._mean_tensor(self.train_raw_z_losses),
            'training/weighted_z_loss': self._mean_tensor(self.train_weighted_z_losses),
            'training/raw_div_loss': self._mean_tensor(self.train_raw_div_losses),
            'training/weighted_div_loss': self._mean_tensor(self.train_weighted_div_losses),
            'training/total_loss': self._mean_tensor(self.train_total_losses),
            'LR_backbone': self.optimizer.param_groups[0]['lr'],
            'LR_head': self.optimizer.param_groups[1]['lr'],
            'LR_router': self.optimizer.param_groups[2]['lr'],
            'temp_logits': torch.tensor(self.model.router.router_temp),
        }

        self.train_seg_losses.clear()
        self.train_aux_losses.clear()
        self.train_raw_z_losses.clear()
        self.train_weighted_z_losses.clear()
        self.train_raw_div_losses.clear()
        self.train_weighted_div_losses.clear()
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
            'validation/raw_z_loss': self._mean_tensor(self.val_raw_z_losses),
            'validation/weighted_z_loss': self._mean_tensor(self.val_weighted_z_losses),
            'validation/raw_div_loss': self._mean_tensor(self.val_raw_div_losses),
            'validation/weighted_div_loss': self._mean_tensor(self.val_weighted_div_losses),
            'validation/total_loss': self._mean_tensor(self.val_total_losses),
            'validation/mIoU': self.metrics["miou"].compute().item() * 100,
            'validation/pixel_acc': self.metrics["pixacc"].compute().item() * 100,
            'validation/mean_acc': self.metrics["macc"].compute().item() * 100,
        }

        self.val_seg_losses.clear()
        self.val_aux_losses.clear()
        self.val_raw_z_losses.clear()
        self.val_weighted_z_losses.clear()
        self.val_raw_div_losses.clear()
        self.val_weighted_div_losses.clear()
        self.val_total_losses.clear()

        log_dict.update(collect_router_metrics('router-val', self.model))

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # 4 groups: backbone, head, router, post_block (all from scratch on CelebA).
        self.backbone_params, self.router_params, self.head_params, self.post_block_params = collect_model_prameters(
            self.model, collect_head=True, collect_post_block=True
        )

        # Head follows the backbone schedule but with its own warmup.
        backbone_fn   = lambda e: backbone_lr_lambda(e, self.warmup_backbone, self.train_epochs)
        head_fn       = lambda e: backbone_lr_lambda(e, self.head_warmup, self.train_epochs)
        router_fn     = lambda e: router_lr_lambda(e, self.router_start_epoch, self.router_warmup, self.train_epochs)
        # post_block follows the backbone schedule; its base lr is already 0.1 * backbone_lr.
        post_block_fn = lambda e: backbone_lr_lambda(e, self.warmup_backbone, self.train_epochs)

        # Group order == lr_lambda order passed to LambdaLR.
        self.optimizer = AdamW(
            [
                {'params': self.backbone_params,   'lr': self.backbone_lr,       'weight_decay': self.weight_decay, 'name': 'backbone'},
                {'params': self.head_params,       'lr': self.head_lr,           'weight_decay': self.weight_decay, 'name': 'head'},
                {'params': self.router_params,     'lr': self.router_lr,         'weight_decay': self.weight_decay,               'name': 'router'},
                {'params': self.post_block_params, 'lr': 0.01 * self.backbone_lr, 'weight_decay': self.weight_decay, 'name': 'post_block'},
            ], betas=self.adam_betas, eps=self.adam_eps
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[backbone_fn, head_fn, router_fn, post_block_fn])

        return [self.optimizer], [self.lr_scheduler]

    def on_after_backward(self):
        # Per-group clipping: the head (from scratch) has its own max_norm.
        clip_gradients(
            backbone_params=self.backbone_params,
            router_params=self.router_params,
            head_params=self.head_params,
            post_block_params=self.post_block_params,
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

    def _freeze_post_block(self):
        for n, p in self.model.named_parameters():
            if 'post_block' in n:
                p.requires_grad_(False)

    def _unfreeze_post_block(self):
        for n, p in self.model.named_parameters():
            if 'post_block' in n:
                p.requires_grad_(True)

    def _freeze_backbone(self):
        # Backbone = ogni parametro che NON è router, head o post_block
        # (stessa partizione di collect_model_prameters / gruppo backbone dell'optimizer).
        for n, p in self.model.named_parameters():
            is_router     = ('router_gate' in n) or ('router' in n) or ('gate' in n)
            is_head       = 'prediction_head' in n
            is_post_block = 'post_block' in n
            if not (is_router or is_head or is_post_block):
                p.requires_grad_(False)

    def _unfreeze_backbone(self):
        # Backbone = ogni parametro che NON è router, head o post_block
        # (stessa partizione di collect_model_prameters / gruppo backbone dell'optimizer).
        for n, p in self.model.named_parameters():
            is_router     = ('router_gate' in n) or ('router' in n) or ('gate' in n)
            is_head       = 'prediction_head' in n
            is_post_block = 'post_block' in n
            if not (is_router or is_head or is_post_block):
                p.requires_grad_(True)
