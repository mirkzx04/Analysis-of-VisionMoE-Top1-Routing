import math

import torch


class MoEAggregator:
    """
    Aggregates routing metrics for the current expert->top-k dispatch.

    With this router, per-expert balance metrics are mostly fixed by construction.
    The useful signals are token coverage, token overlap, and logit sharpness.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: list[int],
        ema_beta: float = 0.9,
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._device = None

        # Per-layer coverage / overlap counters.
        self.token_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.processed_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.multi_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.assignment_sum_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.rel_delta_sum_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.rel_delta_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.expert_effective_rel_delta_sum_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.expert_effective_rel_delta_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]

        self.owner_non_owner_mass_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.owner_total_mass_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.owned_mass_layers = [
            torch.zeros(E, dtype=torch.float64) for E in num_experts
        ]

        # Per-layer dense branch scale counters.
        self.dense_rel_delta_sum_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.dense_metric_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]

        # Per-layer sharpness counters.
        self.logits_std_sum_layers = [0.0 for _ in range(num_layers)]
        self.logits_std_count_layers = [0 for _ in range(num_layers)]
        self.logits_temp_std_sum_layers = [0.0 for _ in range(num_layers)]
        self.logits_temp_std_count_layers = [0 for _ in range(num_layers)]

        self.token_logit_entropy_sum_layers = [0.0 for _ in range(num_layers)]
        self.token_logit_entropy_count_layers = [0 for _ in range(num_layers)]

    @staticmethod
    def _mmm(values):
        """Return (mean, max, min) as floats for a list/tensor; safe for empty inputs."""
        if isinstance(values, (list, tuple)):
            if not values:
                return 0.0, 0.0, 0.0
            t = torch.tensor(values, dtype=torch.float64)
        elif isinstance(values, torch.Tensor):
            if values.numel() == 0:
                return 0.0, 0.0, 0.0
            t = values.to(dtype=torch.float64)
        else:
            return 0.0, 0.0, 0.0
        return float(t.mean().item()), float(t.max().item()), float(t.min().item())

    @staticmethod
    def _ddp_allreduce_(tensor: torch.Tensor):
        """SUM all-reduce if torch.distributed is initialized."""
        if tensor.numel() == 0:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

    def _ensure_device(self, device: torch.device):
        """
        Move all registered tensors to the specified device on first update
        or if device changes between updates.
        """
        if self._device is not None and self._device == device:
            return

        self.token_count_layers = [t.to(device=device) for t in self.token_count_layers]
        self.processed_count_layers = [t.to(device=device) for t in self.processed_count_layers]
        self.multi_count_layers = [t.to(device=device) for t in self.multi_count_layers]
        self.assignment_sum_layers = [t.to(device=device) for t in self.assignment_sum_layers]
        self.rel_delta_sum_layers = [t.to(device=device) for t in self.rel_delta_sum_layers]
        self.rel_delta_count_layers = [t.to(device=device) for t in self.rel_delta_count_layers]
        self.expert_effective_rel_delta_sum_layers = [t.to(device=device) for t in self.expert_effective_rel_delta_sum_layers]
        self.expert_effective_rel_delta_count_layers = [t.to(device=device) for t in self.expert_effective_rel_delta_count_layers]
        self.owner_non_owner_mass_layers = [t.to(device=device) for t in self.owner_non_owner_mass_layers]
        self.owner_total_mass_layers = [t.to(device=device) for t in self.owner_total_mass_layers]
        self.owned_mass_layers = [t.to(device=device) for t in self.owned_mass_layers]
        self.dense_rel_delta_sum_layers = [t.to(device=device) for t in self.dense_rel_delta_sum_layers]

        self.dense_metric_count_layers = [t.to(device=device) for t in self.dense_metric_count_layers]
        self._device = device

    @torch.no_grad()
    def update_layer(
        self,
        layer_idx: int,
        token_idx: torch.Tensor,
        num_tokens: int,
        logits_std: float | None,
        logits_temp_std: float | None,
        logits: torch.Tensor | None,
        expert_idx: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        rel_delta_sum: float | None = None,
        rel_delta_count: int = 0,
        expert_effective_rel_delta_sum: float | None = None,
        expert_effective_rel_delta_count: int = 0,
    ):
        """
        Update accumulators for a single MoE layer on the current batch.

        Args:
            layer_idx: index of the MoE layer.
            token_idx: flattened token ids selected by the router, with repetitions
                when one token is assigned to multiple experts.
            num_tokens: total number of tokens before routing.
            logits_std: std of router raw logits, used to track sharpness.
            logits_temp_std: std of temperature-scaled logits, closer to dispatch sharpness.
            logits: router logits [N, E], used to compute the legacy
                `token_logit_entropy_mean` metric on the expert->token axis.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if self.num_experts[layer_idx] == 0:
            return

        if token_idx.is_cuda:
            device = token_idx.device
        elif logits is not None:
            device = logits.device
        else:
            device = torch.device("cpu")

        self._ensure_device(device)

        N = int(num_tokens)
        token_idx = token_idx.detach()

        if token_idx.numel() > 0:
            assign = torch.bincount(token_idx, minlength=N).to(torch.float64)
        else:
            assign = torch.zeros(N, dtype=torch.float64, device=device)

        processed = assign > 0
        multi = assign > 1

        self.token_count_layers[layer_idx].add_(float(N))
        self.processed_count_layers[layer_idx] += processed.sum().to(torch.float64)
        self.multi_count_layers[layer_idx] += multi.sum().to(torch.float64)
        self.assignment_sum_layers[layer_idx] += assign.sum()

        if logits_std is not None:
            self.logits_std_sum_layers[layer_idx] += float(logits_std)
            self.logits_std_count_layers[layer_idx] += 1

        if logits_temp_std is not None:
            self.logits_temp_std_sum_layers[layer_idx] += float(logits_temp_std)
            self.logits_temp_std_count_layers[layer_idx] += 1

        if logits is not None:
            probs = torch.softmax(logits.detach().to(torch.float32), dim=0)
            num_tokens = probs.shape[0]
            if num_tokens > 1:
                per_expert_entropy = -(probs * (probs + 1e-9).log()).sum(dim=0)
                avg_norm_entropy = (per_expert_entropy / math.log(num_tokens)).mean().item()
            else:
                avg_norm_entropy = 0.0

            self.token_logit_entropy_sum_layers[layer_idx] += float(avg_norm_entropy)
            self.token_logit_entropy_count_layers[layer_idx] += 1

        if rel_delta_sum is not None and rel_delta_count > 0:
            self.rel_delta_sum_layers[layer_idx].add_(float(rel_delta_sum))
            self.rel_delta_count_layers[layer_idx].add_(float(rel_delta_count))

        if expert_effective_rel_delta_sum is not None and expert_effective_rel_delta_count > 0:
            self.expert_effective_rel_delta_sum_layers[layer_idx].add_(float(expert_effective_rel_delta_sum))
            self.expert_effective_rel_delta_count_layers[layer_idx].add_(float(expert_effective_rel_delta_count))

        if expert_idx is not None and weights is not None and token_idx.numel() > 0:
            E = int(self.num_experts[layer_idx])
            token_idx_owner = token_idx.to(device=device, dtype=torch.long)
            expert_idx_owner = expert_idx.detach().to(device=device, dtype=torch.long)
            weights_owner = weights.detach().to(device=device, dtype=torch.float64)

            max_selected_weight = torch.full(
                (N,),
                -float("inf"),
                device=device,
                dtype=weights_owner.dtype,
            )
            max_selected_weight.scatter_reduce_(
                0,
                token_idx_owner,
                weights_owner,
                reduce="amax",
                include_self=True,
            )

            is_owner = weights_owner >= (max_selected_weight[token_idx_owner] - 1e-8)
            total_mass = weights_owner.sum()
            non_owner_mass = weights_owner[~is_owner].sum()

            self.owner_non_owner_mass_layers[layer_idx].add_(non_owner_mass)
            self.owner_total_mass_layers[layer_idx].add_(total_mass)

            if E > 0 and is_owner.any():
                self.owned_mass_layers[layer_idx].index_add_(
                    0,
                    expert_idx_owner[is_owner],
                    weights_owner[is_owner],
                )

    @torch.no_grad()
    def update_dense_layer(
        self,
        layer_idx: int,
        dense_rel_delta: float,
        device: torch.device,
    ):
        """
        Update dense post-MoE scale metrics for a single layer.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if self.num_experts[layer_idx] == 0:
            return

        self._ensure_device(device)

        self.dense_rel_delta_sum_layers[layer_idx].add_(float(dense_rel_delta))
        self.dense_metric_count_layers[layer_idx].add_(1.0)

    @torch.no_grad()
    def finalize(self, include_layer_detail_metrics: bool = True):
        """
        Compute aggregated metrics (per-epoch) and return a compact dict.
        """
        for t in [
            *self.token_count_layers,
            *self.processed_count_layers,
            *self.multi_count_layers,
            *self.assignment_sum_layers,
            *self.rel_delta_sum_layers,
            *self.rel_delta_count_layers,
            *self.expert_effective_rel_delta_sum_layers,
            *self.expert_effective_rel_delta_count_layers,
            *self.owner_non_owner_mass_layers,
            *self.owner_total_mass_layers,
            *self.owned_mass_layers,
            *self.dense_rel_delta_sum_layers,
            *self.dense_metric_count_layers,
        ]:
            self._ddp_allreduce_(t)

        moe_layer_ids = [i for i, E in enumerate(self.num_experts) if E > 0]

        multi_rate_layers = []
        experts_per_processed_token_layers = []
        logits_std_layers = []
        logits_temp_std_layers = []
        token_logit_entropy_layers = []
        non_owner_mass_frac_layers = []
        owner_entropy_norm_layers = []
        expert_raw_rel_delta_layers = []
        expert_effective_rel_delta_layers = []
        dense_rel_delta_layers = []
        dense_to_expert_ratio_layers = []

        total_tokens = 0.0
        total_processed = 0.0

        for layer_idx in moe_layer_ids:
            token_count = float(self.token_count_layers[layer_idx].item())
            processed_count = float(self.processed_count_layers[layer_idx].item())
            multi_count = float(self.multi_count_layers[layer_idx].item())
            assignment_sum = float(self.assignment_sum_layers[layer_idx].item())
            rel_delta_sum = float(self.rel_delta_sum_layers[layer_idx].item())
            rel_delta_count = float(self.rel_delta_count_layers[layer_idx].item())
            expert_effective_rel_delta_sum = float(self.expert_effective_rel_delta_sum_layers[layer_idx].item())
            expert_effective_rel_delta_count = float(self.expert_effective_rel_delta_count_layers[layer_idx].item())
            owner_non_owner_mass = float(self.owner_non_owner_mass_layers[layer_idx].item())
            owner_total_mass = float(self.owner_total_mass_layers[layer_idx].item())
            dense_metric_count = float(self.dense_metric_count_layers[layer_idx].item())

            total_tokens += token_count
            total_processed += processed_count

            if token_count > 0.0:
                multi_rate_layers.append(multi_count / token_count)
            else:
                multi_rate_layers.append(0.0)

            if processed_count > 0.0:
                experts_per_processed_token_layers.append(assignment_sum / processed_count)
            else:
                experts_per_processed_token_layers.append(0.0)

            if self.logits_std_count_layers[layer_idx] > 0:
                avg_std = self.logits_std_sum_layers[layer_idx] / self.logits_std_count_layers[layer_idx]
                logits_std_layers.append(float(avg_std))
            else:
                logits_std_layers.append(0.0)

            if self.logits_temp_std_count_layers[layer_idx] > 0:
                avg_temp_std = (
                    self.logits_temp_std_sum_layers[layer_idx]
                    / self.logits_temp_std_count_layers[layer_idx]
                )
                logits_temp_std_layers.append(float(avg_temp_std))
            else:
                logits_temp_std_layers.append(0.0)

            if self.token_logit_entropy_count_layers[layer_idx] > 0:
                avg_ent = (
                    self.token_logit_entropy_sum_layers[layer_idx]
                    / self.token_logit_entropy_count_layers[layer_idx]
                )
                token_logit_entropy_layers.append(float(avg_ent))
            else:
                token_logit_entropy_layers.append(0.0)

            if rel_delta_count > 0.0:
                expert_raw_rel_delta = rel_delta_sum / rel_delta_count
            else:
                expert_raw_rel_delta = 0.0

            if expert_effective_rel_delta_count > 0.0:
                expert_effective_rel_delta = (
                    expert_effective_rel_delta_sum / expert_effective_rel_delta_count
                )
            else:
                expert_effective_rel_delta = 0.0

            expert_raw_rel_delta_layers.append(expert_raw_rel_delta)
            expert_effective_rel_delta_layers.append(expert_effective_rel_delta)

            if owner_total_mass > 0.0:
                non_owner_mass_frac_layers.append(owner_non_owner_mass / owner_total_mass)
            else:
                non_owner_mass_frac_layers.append(0.0)

            owned_mass = self.owned_mass_layers[layer_idx]
            owned_total = owned_mass.sum()
            E = int(self.num_experts[layer_idx])
            if E > 1 and float(owned_total.item()) > 0.0:
                owner_share = owned_mass / owned_total
                entropy = -(owner_share * (owner_share + 1e-9).log()).sum()
                owner_entropy_norm_layers.append(float((entropy / math.log(E)).item()))
            else:
                owner_entropy_norm_layers.append(0.0)

            if dense_metric_count > 0.0:
                dense_rel_delta = (float(self.dense_rel_delta_sum_layers[layer_idx].item()) / dense_metric_count)
            else : 
                dense_rel_delta = 0.0   
            dense_rel_delta_layers.append(dense_rel_delta)
            dense_to_expert_ratio_layers.append(dense_rel_delta / max(1e-8, expert_effective_rel_delta))
                

        drop_rate = 1.0 - (total_processed / max(1.0, total_tokens))
        multi_assigned_token_rate, _, _ = self._mmm(multi_rate_layers)
        experts_per_processed_token_mean, _, _ = self._mmm(experts_per_processed_token_layers)
        logits_std_mean, _, _ = self._mmm(logits_std_layers)
        logits_temp_std_mean, _, _ = self._mmm(logits_temp_std_layers)
        token_logit_entropy_mean, _, _ = self._mmm(token_logit_entropy_layers)
        non_owner_mass_frac, _, _ = self._mmm(non_owner_mass_frac_layers)
        owner_entropy_norm, _, _ = self._mmm(owner_entropy_norm_layers)
        expert_raw_rel_delta_mean, _, _ = self._mmm(expert_raw_rel_delta_layers)
        expert_effective_rel_delta_mean, _, _ = self._mmm(expert_effective_rel_delta_layers)
        dense_rel_delta_mean, _, _ = self._mmm(dense_rel_delta_layers)
        dense_to_expert_ratio_mean, _, _ = self._mmm(dense_to_expert_ratio_layers)

        metrics = {
            "drop_rate": drop_rate,
            "multi_assigned_token_rate": multi_assigned_token_rate,
            "experts_per_processed_token_mean": experts_per_processed_token_mean,
            "token_logit_entropy_mean": token_logit_entropy_mean,
            "logits_std_mean": logits_std_mean,
            "logits_temp_std_mean": logits_temp_std_mean,
            "non_owner_mass_frac": non_owner_mass_frac,
            "owner_entropy_norm": owner_entropy_norm,
            "expert_raw_rel_delta_mean": expert_raw_rel_delta_mean,
            "expert_effective_rel_delta_mean": expert_effective_rel_delta_mean,
            "dense_rel_delta_mean": dense_rel_delta_mean,
            "dense_to_expert_ratio_mean": dense_to_expert_ratio_mean,
        }
        if include_layer_detail_metrics:
            for moe_layer_idx in range(len(expert_raw_rel_delta_layers)):
            
                metrics[f"moe_layer_{moe_layer_idx}/non_owner_mass_frac"] = float(
                    non_owner_mass_frac_layers[moe_layer_idx]
                )
                metrics[f"moe_layer_{moe_layer_idx}/owner_entropy_norm"] = float(
                    owner_entropy_norm_layers[moe_layer_idx]
                )
                metrics[f"moe_layer_{moe_layer_idx}/expert_raw_rel_delta"] = float(
                    expert_raw_rel_delta_layers[moe_layer_idx]
                )
                metrics[f"moe_layer_{moe_layer_idx}/expert_effective_rel_delta"] = float(
                    expert_effective_rel_delta_layers[moe_layer_idx]
                )
                metrics[f"moe_layer_{moe_layer_idx}/dense_rel_delta"] = float(
                    dense_rel_delta_layers[moe_layer_idx]
                )
                metrics[f"moe_layer_{moe_layer_idx}/dense_to_expert_ratio"] = float(
                    dense_to_expert_ratio_layers[moe_layer_idx]
                )
        return metrics

    def reset(self):
        """Reset all per-epoch accumulators. Keeps the current device setting."""
        self.token_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.token_count_layers
        ]
        self.processed_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.processed_count_layers
        ]
        self.multi_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.multi_count_layers
        ]
        self.assignment_sum_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.assignment_sum_layers
        ]
        self.rel_delta_sum_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.rel_delta_sum_layers
        ]
        self.rel_delta_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.rel_delta_count_layers
        ]
        self.owner_non_owner_mass_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.owner_non_owner_mass_layers
        ]
        self.owner_total_mass_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.owner_total_mass_layers
        ]
        self.owned_mass_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.owned_mass_layers
        ]
        self.dense_rel_delta_sum_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.dense_rel_delta_sum_layers
        ]
        self.dense_metric_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.dense_metric_count_layers
        ]
        self.expert_effective_rel_delta_sum_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.expert_effective_rel_delta_sum_layers
        ]
        self.expert_effective_rel_delta_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.expert_effective_rel_delta_count_layers
        ]

        self.logits_std_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.logits_std_count_layers = [0 for _ in range(self.num_layers)]
        self.logits_temp_std_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.logits_temp_std_count_layers = [0 for _ in range(self.num_layers)]

        self.token_logit_entropy_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.token_logit_entropy_count_layers = [0 for _ in range(self.num_layers)]
