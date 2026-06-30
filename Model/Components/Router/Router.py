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
        sem_weight_temp = 5.0,
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
        gate_out = router_gate(X, positional_features)
        logit_stats = {
            "logits_std": None,
            "logits_temp_std": None,
        }

    
        logits_router = gate_out.to(dtype=torch.float32)
        z_loss = self.z_loss(logits_router)
        # Diversity loss: per-position expert decorrelation, computed in the forward in the
        # same style as z_loss and threaded through every routing branch. It is accumulated
        # across MoE layers in PCENetwork.forward (tot_div_loss), then weighted in the
        # LitModule. Uses the routing probs softmax(logits / router_temp).
        div_loss = self.diversity_loss_from_logits(logits_router, num_positions, N)
        if collect_metrics:
            logit_stats.update({
                "logits_std": logits_router.detach().std(),
                "logits_temp_std": (logits_router / self.router_temp).detach().std(),
            })
            if getattr(router_gate, "is_static_map", False):
                logit_stats["logits_std_pos"] = logit_stats["logits_std"]
                logit_stats["logits_temp_std_pos"] = logit_stats["logits_temp_std"]

        # Route based on current phase (Uniform < uniform_epochs, then specialized).
        if current_epoch is not None and current_epoch < self.uniform_epochs:
            return self._uniform_routing(X, logits_router, logit_stats, z_loss, div_loss)

        # Deterministic fixed position->expert map: activates AFTER the uniform phase
        if getattr(router_gate, "is_deterministic_map", False):
            routing_state = router_gate.build_routing_state(N, X.device)
            return routing_state, z_loss, div_loss, logits_router.detach(), logit_stats

        if current_epoch == None:
            return self._specialized_routing(X, logits_router, logit_stats, z_loss, div_loss, collect_metrics)
        else:
            return self._specialized_routing(X, logits_router, logit_stats, z_loss, div_loss, collect_metrics)

    
    def _specialized_routing(self, X, logits, logit_stats, z_loss, div_loss, collect_metrics=False):
        """
        Specialized top-1 routing (Phase 2/3).

        Standard Mixture-of-Experts (MoE) routing where tokens are assigned to the
        expert with the highest probability (Top-1), subject to capacity constraints.
        """
        N = X.shape[0] # Total numbers of patches
        E = self.num_experts

        # Adding noise in logits (only the selection branch when decoupled)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits.float()) * self.noise_std

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
        return routing_state, z_loss, div_loss, logits_ret.detach(), logit_stats

    def _uniform_routing(self, X, logits, logit_stats, z_loss, div_loss):
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

        logits_ret = logits[0] if isinstance(logits, tuple) else logits
        return routing_state, z_loss, div_loss, logits_ret.detach(), logit_stats

    def z_loss(self, logits):
        """
        Computes Z-loss to encourage smaller logits.

        Args:
            logits (torch.Tensor): Router logits.
        """

        return torch.logsumexp(logits, dim = -1).square().mean()

    def diversity_loss_from_logits(self, logits, num_positions, N):
        """Routing-prob diversity loss from raw logits. Returns a 0-d tensor.
        Guards the [B, P, E] reshape: needs a valid num_positions dividing N."""
        if not num_positions or num_positions <= 0 or (N % num_positions) != 0:
            return logits.new_tensor(0.0)
        probs = F.softmax(logits / self.router_temp, dim=-1)
        return self.diversity_loss(probs, N // num_positions, num_positions)

    def diversity_loss(self, probs, batch_szie, num_positions) :
        E = probs.shape[-1]
        if E <= 1 : 
            return probs.new_tensor(0.0)

        probs = probs.reshape(batch_szie, num_positions, E)
        probs = probs.permute(1, 0, 2)

        probs_norm = F.normalize(probs, p = 2, dim = 1, eps = 1e-9)
        corr = torch.einsum(
            "pbe,pbf->pef",
            probs_norm,
            probs_norm,
        )

        eye = torch.eye(E, device=probs.device, dtype = probs.dtype).unsqueeze(0)
        off_diagonal = corr * (1.0 - eye)
        return (off_diagonal.square().sum(dim = (-2, -1)) / (E * (E - 1))).mean()