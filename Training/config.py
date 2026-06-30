"""Centralised training config in five dataclasses:

  - ``HParams``           : architecture, optimizer, LR/temperature schedule, grad clipping.
  - ``RouterTypesParams`` : router-type selectors (which gate each PCELayer builds).
  - ``LossWeights``       : aux-loss weights and CE config.
  - ``AugParams``         : data augmentation (mixup / cutmix).
  - ``EpochParams``       : epoch counts and warmups (schedule durations).
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class HParams:
    # --- Model architecture ---
    num_experts: int = 16
    layer_number: int = 8
    patch_size: int = 16
    halo_for_patches: int = 2
    input_size: int = 224
    capacity_factor_train: float = 2.0
    capacity_factor_val: float = 2.0
    task: str = "seg"
    num_classes: int = 21

    # --- Optimizer / training loop ---
    batch_size: int = 64
    weight_decay: float = 1e-2
    backbone_lr: float = 2e-5
    head_lr: float = 1e-3
    router_lr: float = 1e-3
    accumulate_grad_batches: int = 2
    adam_betas: Tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-8

    # --- Temperature schedule ---
    temp_init: float = 1.75
    temp_mid: float = 1.2
    temp_final: float = 0.5

    # --- Gradient clipping (per-group max norms) ---
    backbone_max_norm: float = 1.5
    router_max_norm: float = 0.5
    head_max_norm: float = 1.0

    @property
    def lr(self) -> float:
        """Alias for ``backbone_lr`` (Tiny called it ``lr``)."""
        return self.backbone_lr


@dataclass
class RouterTypesParams:
    """Router-type selectors: which gate each PCELayer builds.

    Separated from HParams so the router variant is configured independently of
    architecture / optimizer hyperparameters. The four selectors below are
    mutually exclusive (at most one True); all-False builds the default
    dual-branch RouterGate. ``unified_router`` is a sub-mode of that default
    RouterGate and is only honoured when the four selectors are all False.
    """
    use_static_map: bool = False
    pos_only: bool = False
    semantic_only: bool = False
    interaction: bool = False
    unified_router: bool = False

    # InteractionGate knobs (only used when interaction=True).
    interaction_hidden_size: int = 128
    interaction_include_main_effects: bool = False

    def __post_init__(self):
        exclusive = (self.use_static_map, self.pos_only, self.semantic_only, self.interaction)
        if sum(bool(f) for f in exclusive) > 1:
            raise ValueError(
                "RouterTypesParams: at most one of use_static_map / pos_only / "
                "semantic_only / interaction can be True "
                f"(got use_static_map={self.use_static_map}, pos_only={self.pos_only}, "
                f"semantic_only={self.semantic_only}, interaction={self.interaction})."
            )


@dataclass
class LossWeights:
    z_loss_weight: float = 1e-2
    # Per-position expert-decorrelation diversity loss. 0.0 -> OFF (training unchanged).
    # ~0.3-0.5 gave strong semantic specialization in "DIVERSITY LOSS TEST".
    # NB: only CelebaLitModule reads this field; Pascal/Tiny ignore it.
    diversity_loss_weight: float = 0.3
    label_smoothing: float = 0.0
    ignore_index: int = 255


@dataclass
class AugParams:
    """Data augmentation (mixup / cutmix), historically Tiny-only."""
    use_mixup_cutmix: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.3


@dataclass
class EpochParams:
    """Epoch counts and warmups (schedule durations)."""
    train_epochs: int = 60
    # router_start_epoch is derived in the lit modules as = uniform_epochs
    # (router turns on when the uniform/warmup phase ends), not a hyperparameter.
    uniform_epochs: int = 0
    warmup_backbone: int = 5
    head_warmup: int = 10
    router_warmup: int = 5
    temp_epochs: int = 25
