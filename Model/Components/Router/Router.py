import math

import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass

@dataclass(slots=True)
class RoutingState:
    token_idx: torch.Tensor
    expert_idx: torch.Tensor
    slot_idx: torch.Tensor
    weights: torch.Tensor
    num_tokens: int
    num_experts: int
    capacity: int

class Router(nn.Module):
    def __init__(
        self,
        num_experts,
        num_layers,
        router_temp = 1.5,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 1.50,
        sem_weight_temp = 1.0,
        uniform_epochs = 0,
        ):
        super().__init__()
        """
        Router class with staged training support.

        Args:
            num_experts (int): Number of experts.
            num_layers (int): Number of layers.
            noise_epsilon (float): Epsilon for noise stability.
            router_temp (float): Temperature for the router logits.
            sem_weight_temp (float): Temperature of the semantic weighting softmax,
            renormalized per token over the experts that selected it (S(n)).
            capacity_factor_train (float): Capacity factor during training.
            capacity_factor_eval (float): Capacity factor during evaluation.
            noise_std (float): Standard deviation for the noise added to logits.
            Ccap (float): Capacity coefficient (unused in current logic, but kept for compatibility).
        """
        self.num_experts = num_experts
        self.num_layers = num_layers

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.noise_std = 0

        self.router_temp = router_temp
        self.sem_weight_temp = sem_weight_temp
        self.uniform_epochs = uniform_epochs


    def forward(
        self,
        X,
        router_gate,
        positional_features,
        num_positions = None,
        current_epoch = None,
        h_patches = None,
        w_patches = None,
        unified_router = False,
        collect_metrics = False,):
        """
        Router forward pass with automatic phase management based on epochs.

        Args:
            X (torch.Tensor): Input tensor of shape [N, C, H, W] (or flattened [B*P, ...]).
            router_gate (nn.Module): The gate module to compute logits.
            current_epoch (int, optional): Current training epoch to determine routing phase.

        Returns:
            dispatch (torch.Tensor): Dispatch tensor for routing.
            combine (torch.Tensor): Combine tensor for aggregating results.
            z_loss (torch.Tensor): Router z-loss value.
            aux_loss (torch.Tensor): Auxiliary load balancing loss.
            logits (torch.Tensor): Raw logits (detached, on CPU).
            logit_stats (dict): Optional router std metrics, including the
                selection logits and branch-level positional/semantic logits.
        """

        N = X.shape[0] # Total number of patches
        E = self.num_experts
        unified_router = bool(unified_router or getattr(router_gate, "unified_router", False))

        # Always compute logits for monitoring and potential use.
        # RouterGate returns separated semantic/positional logits only when the
        # positional branch exists independently; unified/static gates return
        # already-combined logits.
        gate_out = router_gate(X, positional_features)
        spatial_loss = X.new_tensor(0.0, dtype=torch.float32)
        logit_stats = {
            "logits_std": None,
            "logits_temp_std": None,
            "logits_std_pos": None,
            "logits_std_sem": None,
            "logits_temp_std_pos": None,
            "logits_temp_std_sem": None,
        }

        if isinstance(gate_out, tuple):
            sem_logits, pos_logits = gate_out
            sem_logits, pos_logits = sem_logits.float(), pos_logits.float()

            # Selection by position; semantics weights the chosen expert's output
            # downstream in _specialized_routing.
            weighting_logits = sem_logits
            selection_logits = pos_logits

            logits_router = (selection_logits, weighting_logits)
            if ( self.training and not getattr(router_gate, "is_static_map", False) and not unified_router and h_patches is not None and w_patches is not None):
                spatial_loss = self.spatial_loss(selection_logits, h_patches, w_patches)
            z_loss = self.z_loss(weighting_logits)

            if collect_metrics:
                logit_stats.update({
                    "logits_std_pos": pos_logits.detach().std(),
                    "logits_std_sem": sem_logits.detach().std(),
                    "logits_temp_std_pos": (pos_logits / self.router_temp).detach().std(),
                    "logits_temp_std_sem": (sem_logits / self.router_temp).detach().std(),
                })
        else:
            logits_router = gate_out.to(dtype=torch.float32)
            z_loss = self.z_loss(logits_router)
            # spatial_loss = self.spatial_loss(pos_logits, h_patches, w_patches)
            if collect_metrics:
                logit_stats.update({
                    "logits_std": logits_router.detach().std(),
                    "logits_temp_std": (logits_router / self.router_temp).detach().std(),
                })
                if getattr(router_gate, "is_static_map", False):
                    logit_stats["logits_std_pos"] = logit_stats["logits_std"]
                    logit_stats["logits_temp_std_pos"] = logit_stats["logits_temp_std"]

        # Route based on current phase (Uniform < 30 epochs, Specialized >= 30 epochs)
        if current_epoch == None:
            return self._specialized_routing(X, logits_router, logit_stats, z_loss, spatial_loss, collect_metrics)
        if current_epoch < self.uniform_epochs:
            return self._uniform_routing(X, logits_router, logit_stats, z_loss, spatial_loss)
        if getattr(router_gate, "is_static_map", False):
            return self._static_expert_region_routing(X, logits_router, logit_stats, z_loss, spatial_loss, num_positions)
        else:
            return self._specialized_routing(X, logits_router, logit_stats, z_loss, spatial_loss, collect_metrics)

    def _static_expert_region_routing(
        self,
        X,
        logits,
        logit_stats,
        z_loss,
        spatial_loss,
        num_positions,
    ):
        """
        Static expert -> region routing.

        X: [B*P, C, H, W]
        logits: [B*P, E], prodotti da StaticMapGate, quindi dipendono solo dalla posizione
        num_positions: P, numero di posizioni/patch per immagine
        """

        N = X.shape[0]
        E = self.num_experts
        P = int(num_positions)

        if N % P != 0:
            raise ValueError(f"N={N} must be divisible by num_positions={P}")

        B = N // P

        # [B*P, E] -> [B, P, E]
        logits_bpe = logits.view(B, P, E)

        # Siccome il gate è statico, ogni immagine dovrebbe avere la stessa mappa.
        # Uso mean(0) invece di [0] per evitare dipendenze spurie dalla prima immagine.
        logits_pos = logits_bpe.mean(dim=0)  # [P, E]

        probs_pos = F.softmax(logits_pos.float() / self.router_temp, dim=-1)  # [P, E]

        # Capacità per esperto in termini di posizioni per immagine
        cap_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        k_pos = max(1, math.ceil(cap_factor * P / E))
        k_pos = min(k_pos, P)

        # Expert-choice statico:
        # ogni esperto sceglie le sue k_pos posizioni preferite
        topk_prob, topk_pos = torch.topk(
            probs_pos,
            k=k_pos,
            dim=0,
            largest=True,
            sorted=True,
        )  # [k_pos, E]

        # Replica la stessa mappa per ogni immagine del batch
        batch_offsets = torch.arange(B, device=X.device).view(B, 1, 1) * P
        token_idx = (
            topk_pos.view(1, k_pos, E)
            .add(batch_offsets)
            .permute(2, 0, 1)
            .reshape(E, B * k_pos)
        )

        expert_idx = torch.arange(E, device=X.device).view(E, 1).expand(E, B * k_pos)
        slot_idx = torch.arange(B * k_pos, device=X.device).view(1, B * k_pos).expand(E, B * k_pos)
        weights = (
            topk_prob.view(1, k_pos, E)
            .expand(B, k_pos, E)
            .permute(2, 0, 1)
            .reshape(E, B * k_pos)
            .float()
        )

        routing_state = RoutingState(
            token_idx=token_idx.reshape(-1),
            expert_idx=expert_idx.reshape(-1),
            slot_idx=slot_idx.reshape(-1),
            weights=weights.reshape(-1),
            num_experts=E,
            num_tokens=N,
            capacity=B * k_pos,
        )

        return routing_state, spatial_loss, z_loss, logits.detach(), logit_stats

    def _specialized_routing(self, X, logits, logit_stats, z_loss, spatial_loss, collect_metrics=False):
        """
        Specialized top-1 routing (Phase 2/3).

        Standard Mixture-of-Experts (MoE) routing where tokens are assigned to the
        expert with the highest probability (Top-1), subject to capacity constraints.
        """
        N = X.shape[0] # Total numbers of patches
        E = self.num_experts

        # Adding noise in logits (only the selection branch when decoupled)
        if self.training and self.noise_std > 0:
            if isinstance(logits, tuple):
                logits = (logits[0] + torch.randn_like(logits[0]) * self.noise_std,) + tuple(logits[1:])
            else:
                logits = logits + torch.randn_like(logits.float()) * self.noise_std

        if isinstance(logits, tuple) :
            probs_e2t = F.softmax(logits[0] / self.router_temp, dim = -1)
        else :
            probs_e2t = F.softmax(logits.float() / self.router_temp, dim = -1) # [N, E]

        # Calculate capacity per expert (must be an integer)
        cap_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        ccap = math.ceil(cap_factor * N / E)
        k = min(max(1, int(ccap)), N)

        # Extract topk probs and topk index
        topk_prob, topk_idx = torch.topk(
            probs_e2t,
            k = k,
            dim = 0,
            largest=True,
            sorted=True,
        )

        token_idx = topk_idx.transpose(0, 1).contiguous()
        expert_idx = torch.arange(E, device=X.device).view(E, 1).expand(E, k)
        slot_idx = torch.arange(k, device=X.device).view(1, k).expand(E, k)

        if isinstance(logits, tuple):
            # --- Stage 2: semantic weighting renormalized over S(n) = {experts that selected token n} ---
            sem_logits = logits[1]                                # [N, E]
            sel_sem_logit = sem_logits.gather(0, topk_idx)
            flat_tok = topk_idx.reshape(-1)

            # Per-token max over S(n) for numerical stability (sem logits std is large/rising).
            with torch.no_grad():
                max_sem = torch.full(
                    (N,), float("-inf"), device=X.device, dtype=sel_sem_logit.dtype
                )
                max_sem.scatter_reduce_(
                    0, flat_tok, sel_sem_logit.reshape(-1), reduce="amax", include_self=True
                )

            num = torch.exp((sel_sem_logit - max_sem[topk_idx]) / self.sem_weight_temp)  # [k, E]
            denom = torch.zeros(N, device=X.device, dtype=num.dtype).index_add(
                0, flat_tok, num.reshape(-1)
            )                                                     # denom[n] = sum_{e' in S(n)} exp(...)
            w_sem = num / (denom[topk_idx] + 1e-9)                # [k, E] convex over S(n)

            p_sel = F.softmax(logits[0] / self.router_temp, dim=-1).gather(0, topk_idx)  # [k, E]
            p_sel_st = 1.0 + p_sel - p_sel.detach()

            weights = (w_sem * p_sel_st).transpose(0, 1).contiguous().float()  # [E, k]
        else:
            weights = topk_prob.transpose(0, 1).contiguous().float()
        routing_state = RoutingState(
            token_idx=token_idx.reshape(-1),
            expert_idx=expert_idx.reshape(-1),
            slot_idx=slot_idx.reshape(-1),
            weights=weights.reshape(-1),
            num_experts=E,
            num_tokens=N,
            capacity=k
        )

        logits_ret = logits[0] if isinstance(logits, tuple) else logits
        return routing_state, spatial_loss, z_loss, logits_ret.detach(), logit_stats

    def _uniform_routing(self, X, logits, logit_stats, z_loss, spatial_loss):
        """
        Uniform routing (Phase 1).

        Distributes tokens evenly across all experts without using router decisions.
        This allows experts to learn diverse features before routing specialization.
        """
        N = X.shape[0]
        E = self.num_experts

        cap_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        ccap = int(max(1, math.ceil(cap_factor * N / E)))

        token_idx = torch.arange(N, device=X.device)
        expert_idx = token_idx.remainder(E)
        slot_idx = torch.div(token_idx, E, rounding_mode="floor")

        keep = slot_idx < ccap

        kept_token_idx = token_idx[keep]
        kept_expert_idx = expert_idx[keep]
        kept_slot_idx = slot_idx[keep]

        token_table = torch.zeros(E, ccap, device=X.device, dtype=torch.long)
        weight_table = torch.zeros(E, ccap, device=X.device, dtype=torch.float32)

        flat_slot = kept_expert_idx * ccap + kept_slot_idx
        token_table.view(-1).scatter_(0, flat_slot, kept_token_idx)
        weight_table.view(-1).scatter_(
            0,
            flat_slot,
            torch.ones_like(kept_token_idx, dtype=torch.float32),
        )

        full_expert_idx = torch.arange(E, device=X.device).view(E, 1).expand(E, ccap)
        full_slot_idx = torch.arange(ccap, device=X.device).view(1, ccap).expand(E, ccap)

        routing_state = RoutingState(
            token_idx=token_table.reshape(-1),
            expert_idx=full_expert_idx.reshape(-1),
            slot_idx=full_slot_idx.reshape(-1),
            weights=weight_table.reshape(-1),
            num_tokens=N,
            num_experts=E,
            capacity=ccap
        )

        spatial_loss = X.new_tensor(0.0, dtype=torch.float32)

        logits_ret = logits[0] if isinstance(logits, tuple) else logits
        return routing_state, spatial_loss, z_loss, logits_ret.detach(), logit_stats

    def z_loss(self, logits):
        """
        Computes Z-loss to encourage smaller logits.

        Args:
            logits (torch.Tensor): Router logits.
        """

        return torch.logsumexp(logits, dim = -1).square().mean()

    def spatial_loss(self, pos_logits, h_patches, w_patches):
        """
        Positional specialization loss (centroidi).

        Spinge gli expert su regioni spaziali separate: massimizza la distanza
        media tra i centroidi (sulla griglia) delle distribuzioni posizione-per-expert.

        Args:
            pos_logits (torch.Tensor): logit posizionali [B*P, E].
            h_patches, w_patches (int): griglia delle posizioni per immagine.
        """
        N, E = pos_logits.shape
        P = h_patches * w_patches

        if P <= 1 or E <= 1 or N % P != 0:
            return pos_logits.new_tensor(0.0)

        B = N // P

        # [B*P, E] -> [B, P, E] -> [P, E]
        pos_map = pos_logits.view(B, P, E).mean(dim=0)

        # distribuzione posizione-per-esperto
        w = F.softmax(pos_map / self.router_temp, dim=0)  # [P, E]

        device = pos_logits.device
        rows = torch.arange(h_patches, device=device, dtype=torch.float32) / max(h_patches - 1, 1)
        cols = torch.arange(w_patches, device=device, dtype=torch.float32) / max(w_patches - 1, 1)

        grid_r = rows[:, None].expand(h_patches, w_patches).reshape(P)
        grid_c = cols[None, :].expand(h_patches, w_patches).reshape(P)
        pos_2d = torch.stack([grid_r, grid_c], dim=1)  # [P, 2]

        centroids = w.transpose(0, 1) @ pos_2d  # [E, 2]

        dists = torch.cdist(centroids, centroids, p=2.0)
        off_diag = ~torch.eye(E, dtype=torch.bool, device=device)

        return -dists[off_diag].mean()
