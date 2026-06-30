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

        # Per-layer sharpness counters. The *_sum_layers accumulate on-device
        # (float64) so per-step updates need no GPU->CPU sync; they are read once
        # per epoch in finalize(). The *_count_layers stay python ints (incremented
        # without touching any tensor).
        self.logits_std_sum_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.logits_std_count_layers = [0 for _ in range(num_layers)]
        self.logits_temp_std_sum_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.logits_temp_std_count_layers = [0 for _ in range(num_layers)]

        self.token_logit_entropy_sum_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.token_logit_entropy_count_layers = [0 for _ in range(num_layers)]

        # Per-layer expert x position joint counts (lazily allocated to [E, P]).
        self.expert_pos_joint_layers = [None for _ in range(num_layers)]
        # Per-layer expert x class joint counts (lazily allocated to [E, C]).
        self.expert_class_joint_layers = [None for _ in range(num_layers)]

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
    def _mutual_information(joint):
        """MI in nats between two discrete variables from a joint count matrix."""
        total = joint.sum()
        if float(total.item()) <= 0.0:
            return 0.0
        p = joint / total
        p_row = p.sum(dim=1, keepdim=True)
        p_col = p.sum(dim=0, keepdim=True)
        nz = p > 0
        ratio = p[nz] / (p_row.expand_as(p)[nz] * p_col.expand_as(p)[nz])
        return float((p[nz] * ratio.log()).sum().item())

    @staticmethod
    def _expert_pos_entropies(joint):
        """
        Normalized marginal/conditional entropies from an [E, P] joint count matrix,
        both in [0, 1] (divided by log E). Computed on the HARD dispatch, so non
        sem-confused:
            H(e)      -> bilanciamento uso expert (alto = bilanciato)
            H(e|pos)  -> specializzazione per posizione (basso = confident)
        Nota: MI_norm = H(e)_norm - H(e|pos)_norm.
        Returns (marg_norm, cond_norm).
        """
        total = joint.sum()
        E = joint.shape[0]
        if float(total.item()) <= 0.0 or E <= 1:
            return 0.0, 0.0
        log_e = math.log(E)
        p = joint / total                                  # p(e, pos)
        p_e = p.sum(dim=1)                                  # p(e)   [E]
        p_pos = p.sum(dim=0)                                # p(pos) [P]
        nz_e = p_e > 0
        h_marg = float(-(p_e[nz_e] * p_e[nz_e].log()).sum().item())
        q = p / p_pos.clamp_min(1e-12).unsqueeze(0)        # p(e|pos)
        nz = p > 0
        h_cond = float(-(p[nz] * q[nz].log()).sum().item())  # H(e|pos)
        return h_marg / log_e, h_cond / log_e

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
        self.logits_std_sum_layers = [t.to(device=device) for t in self.logits_std_sum_layers]
        self.logits_temp_std_sum_layers = [t.to(device=device) for t in self.logits_temp_std_sum_layers]
        self.token_logit_entropy_sum_layers = [t.to(device=device) for t in self.token_logit_entropy_sum_layers]
        self.expert_pos_joint_layers = [
            None if t is None else t.to(device=device) for t in self.expert_pos_joint_layers
        ]
        self.expert_class_joint_layers = [
            None if t is None else t.to(device=device) for t in self.expert_class_joint_layers
        ]
        self._device = device

    @torch.no_grad()
    def update_layer(
        self,
        layer_idx: int,
        token_idx: torch.Tensor,
        num_tokens: int,
        logit_stats: dict[str, float | None] | None = None,
        logits: torch.Tensor | None = None,
        expert_idx: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        rel_delta_sum: float | None = None,
        rel_delta_count: int = 0,
        expert_effective_rel_delta_sum: float | None = None,
        expert_effective_rel_delta_count: int = 0,
        num_positions: int | None = None,
        class_labels: torch.Tensor | None = None,
        num_classes: int | None = None,
        patch_class: torch.Tensor | None = None,
    ):
        """
        Update accumulators for a single MoE layer on the current batch.

        Args:
            layer_idx: index of the MoE layer.
            token_idx: flattened token ids selected by the router, with repetitions
                when one token is assigned to multiple experts.
            num_tokens: total number of tokens before routing.
            logit_stats: router logit statistics emitted by Router.forward.
            logits: router logits [N, E], used to compute the
                `token_logit_entropy_mean` metric: the per-token entropy of the
                softmax over experts (dim=-1), normalized by log(E). Low = the
                router is confident in its per-token expert choice.
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

        logit_stats = logit_stats or {}

        logits_std = logit_stats.get("logits_std")
        if logits_std is not None:
            self.logits_std_sum_layers[layer_idx].add_(logits_std)
            self.logits_std_count_layers[layer_idx] += 1

        logits_temp_std = logit_stats.get("logits_temp_std")
        if logits_temp_std is not None:
            self.logits_temp_std_sum_layers[layer_idx].add_(logits_temp_std)
            self.logits_temp_std_count_layers[layer_idx] += 1

        if logits is not None:
            # Per-token routing confidence: entropy of the softmax over EXPERTS
            # (dim=-1) for each token, normalized by log(E). Low = the router is
            # confident in its per-token expert choice; high = ~uniform over experts.
            probs = torch.softmax(logits.detach().to(torch.float32), dim=-1)
            E_logits = probs.shape[-1]
            if E_logits > 1:
                per_token_entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
                avg_norm_entropy = (per_token_entropy / math.log(E_logits)).mean()
                # add on-device (kept as a tensor, no .item()) so no per-step
                # GPU->CPU sync; the E_logits <= 1 case contributes exactly 0.
                self.token_logit_entropy_sum_layers[layer_idx].add_(avg_norm_entropy)
            self.token_logit_entropy_count_layers[layer_idx] += 1

        # Accept either python scalars or on-device tensors; tensors are added in
        # place without .item()/float() so no per-step sync occurs. Adding a zero
        # sum/count is a no-op, so dropping the previous `count > 0` guard does not
        # change the accumulated values.
        if rel_delta_sum is not None:
            self.rel_delta_sum_layers[layer_idx].add_(
                rel_delta_sum if torch.is_tensor(rel_delta_sum) else float(rel_delta_sum)
            )
            self.rel_delta_count_layers[layer_idx].add_(
                rel_delta_count if torch.is_tensor(rel_delta_count) else float(rel_delta_count)
            )

        if expert_effective_rel_delta_sum is not None:
            self.expert_effective_rel_delta_sum_layers[layer_idx].add_(
                expert_effective_rel_delta_sum if torch.is_tensor(expert_effective_rel_delta_sum)
                else float(expert_effective_rel_delta_sum)
            )
            self.expert_effective_rel_delta_count_layers[layer_idx].add_(
                expert_effective_rel_delta_count if torch.is_tensor(expert_effective_rel_delta_count)
                else float(expert_effective_rel_delta_count)
            )

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
            # Mask instead of boolean-indexing to avoid the data-dependent output
            # shape (which forces a GPU->CPU sync). Masked-out terms contribute
            # exactly 0, so the sums are unchanged.
            is_owner_f = is_owner.to(weights_owner.dtype)
            non_owner_mass = (weights_owner * (1.0 - is_owner_f)).sum()

            self.owner_non_owner_mass_layers[layer_idx].add_(non_owner_mass)
            self.owner_total_mass_layers[layer_idx].add_(total_mass)

            if E > 0:
                # index_add over all tokens with owner-masked weights is identical to
                # adding only the owners' weights (non-owners add 0) but needs no sync.
                self.owned_mass_layers[layer_idx].index_add_(
                    0,
                    expert_idx_owner,
                    weights_owner * is_owner_f,
                )

        # Expert x position joint counts (position = token_idx % num_positions).
        if num_positions is not None and expert_idx is not None and token_idx.numel() > 0:
            E = int(self.num_experts[layer_idx])
            P = int(num_positions)
            if self.expert_pos_joint_layers[layer_idx] is None:
                self.expert_pos_joint_layers[layer_idx] = torch.zeros(E, P, dtype=torch.float64, device=device)
            expert_long = expert_idx.detach().to(device=device, dtype=torch.long)
            pos = token_idx.to(device=device, dtype=torch.long) % P
            flat = expert_long * P + pos
            self.expert_pos_joint_layers[layer_idx].view(-1).index_add_(
                0, flat, torch.ones_like(flat, dtype=torch.float64)
            )

        # Expert x class joint counts. For tokenized image patches, token_idx // P
        # maps each dispatched token back to its image label.
        if (
            num_positions is not None
            and class_labels is not None
            and num_classes is not None
            and expert_idx is not None
            and token_idx.numel() > 0
        ):
            E = int(self.num_experts[layer_idx])
            P = int(num_positions)
            C = int(num_classes)
            labels = class_labels.detach()
            if labels.dim() == 2 and labels.shape[-1] == C:
                labels = labels.argmax(dim=1)
            if P > 0 and C > 0 and labels.dim() == 1 and labels.numel() > 0:
                if self.expert_class_joint_layers[layer_idx] is None:
                    self.expert_class_joint_layers[layer_idx] = torch.zeros(
                        E, C, dtype=torch.float64, device=device
                    )
                expert_long = expert_idx.detach().to(device=device, dtype=torch.long)
                token_long = token_idx.to(device=device, dtype=torch.long)
                labels_long = labels.to(device=device, dtype=torch.long)
                image_idx = token_long // P
                safe_image_idx = image_idx.clamp(min=0, max=labels_long.numel() - 1)
                cls = labels_long[safe_image_idx]
                valid = (
                    (image_idx >= 0)
                    & (image_idx < labels_long.numel())
                    & (cls >= 0)
                    & (cls < C)
                    & (expert_long >= 0)
                    & (expert_long < E)
                )
                flat = expert_long.clamp(min=0, max=E - 1) * C + cls.clamp(min=0, max=C - 1)
                self.expert_class_joint_layers[layer_idx].view(-1).index_add_(
                    0, flat, valid.to(dtype=torch.float64)
                )
        # Expert x class joint counts for SEGMENTATION. `patch_class` is a
        # per-(image, position) dominant-class map flattened to [B*P] row-major
        # (see train_utils.compute_patch_class). Unlike the classification branch
        # above, the token->class mapping is direct: token_idx already indexes the
        # B*P space, so patch_class_flat[token_idx] is the class of each dispatched
        # token's patch. Void/empty cells carry an id >= num_classes and are
        # dropped by the `cls < C` mask, matching the MI(pos,class) ceiling.
        if (
            patch_class is not None
            and num_classes is not None
            and expert_idx is not None
            and token_idx.numel() > 0
        ):
            E = int(self.num_experts[layer_idx])
            C = int(num_classes)
            patch_class_flat = patch_class.detach().to(device=device, dtype=torch.long).reshape(-1)
            # patch_class must cover this layer's full B*P token space; skip if a
            # layer's grid ever differs from the map we were handed.
            if patch_class_flat.numel() == N:
                if self.expert_class_joint_layers[layer_idx] is None:
                    self.expert_class_joint_layers[layer_idx] = torch.zeros(
                        E, C, dtype=torch.float64, device=device
                    )
                tok = token_idx.to(device=device, dtype=torch.long)
                expert_long = expert_idx.detach().to(device=device, dtype=torch.long)
                safe_tok = tok.clamp(min=0, max=patch_class_flat.numel() - 1)
                cls = patch_class_flat[safe_tok]
                valid = (
                    (tok >= 0)
                    & (tok < patch_class_flat.numel())
                    & (cls >= 0)
                    & (cls < C)
                    & (expert_long >= 0)
                    & (expert_long < E)
                )
                flat = (
                    expert_long.clamp(min=0, max=E - 1) * C + cls.clamp(min=0, max=C - 1)
                )
                self.expert_class_joint_layers[layer_idx].view(-1).index_add_(
                    0, flat, valid.to(dtype=torch.float64)
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

        self.dense_rel_delta_sum_layers[layer_idx].add_(
            dense_rel_delta if torch.is_tensor(dense_rel_delta) else float(dense_rel_delta)
        )
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

        for t in self.expert_pos_joint_layers:
            if t is not None:
                self._ddp_allreduce_(t)
        for t in self.expert_class_joint_layers:
            if t is not None:
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
        mi_expert_pos_layers = []
        mi_expert_class_layers = []
        marg_entropy_expert_layers = []
        cond_entropy_expert_pos_layers = []

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

            pos_joint = self.expert_pos_joint_layers[layer_idx]
            if pos_joint is not None:
                mi_expert_pos_layers.append(self._mutual_information(pos_joint))
                marg_norm, cond_norm = self._expert_pos_entropies(pos_joint)
                marg_entropy_expert_layers.append(marg_norm)
                cond_entropy_expert_pos_layers.append(cond_norm)
            else:
                mi_expert_pos_layers.append(0.0)
                marg_entropy_expert_layers.append(0.0)
                cond_entropy_expert_pos_layers.append(0.0)

            class_joint = self.expert_class_joint_layers[layer_idx]
            if class_joint is not None:
                mi_c = self._mutual_information(class_joint)
                mi_expert_class_layers.append(mi_c)
            else:
                mi_expert_class_layers.append(0.0)

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
        mi_expert_position_mean, _, _ = self._mmm(mi_expert_pos_layers)
        mi_expert_class_mean, _, _ = self._mmm(mi_expert_class_layers)
        marg_entropy_expert_mean, _, _ = self._mmm(marg_entropy_expert_layers)
        cond_entropy_expert_pos_mean, _, _ = self._mmm(cond_entropy_expert_pos_layers)

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
            "mi_expert_position": mi_expert_position_mean,
            "mi_expert_class": mi_expert_class_mean,
            "marg_entropy_expert": marg_entropy_expert_mean,
            "cond_entropy_expert_pos": cond_entropy_expert_pos_mean,
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
                metrics[f"moe_layer_{moe_layer_idx}/mi_expert_position"] = float(
                    mi_expert_pos_layers[moe_layer_idx]
                )
                metrics[f"moe_layer_{moe_layer_idx}/mi_expert_class"] = float(
                    mi_expert_class_layers[moe_layer_idx]
                )
                metrics[f"moe_layer_{moe_layer_idx}/marg_entropy_expert"] = float(
                    marg_entropy_expert_layers[moe_layer_idx]
                )
                metrics[f"moe_layer_{moe_layer_idx}/cond_entropy_expert_pos"] = float(
                    cond_entropy_expert_pos_layers[moe_layer_idx]
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

        self.logits_std_sum_layers = [torch.zeros_like(t) for t in self.logits_std_sum_layers]
        self.logits_std_count_layers = [0 for _ in range(self.num_layers)]
        self.logits_temp_std_sum_layers = [torch.zeros_like(t) for t in self.logits_temp_std_sum_layers]
        self.logits_temp_std_count_layers = [0 for _ in range(self.num_layers)]

        self.token_logit_entropy_sum_layers = [torch.zeros_like(t) for t in self.token_logit_entropy_sum_layers]
        self.token_logit_entropy_count_layers = [0 for _ in range(self.num_layers)]

        self.expert_pos_joint_layers = [None for _ in range(self.num_layers)]
        self.expert_class_joint_layers = [None for _ in range(self.num_layers)]
