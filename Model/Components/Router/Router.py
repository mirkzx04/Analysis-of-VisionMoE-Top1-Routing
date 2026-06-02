import math

import torch
from torch.distributions import OneHotCategorical
import torch.nn.functional as F

from torch import logit, nn
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
        ):
        super().__init__()
        """
        Router class with staged training support.

        Args:
            num_experts (int): Number of experts.
            num_layers (int): Number of layers.
            noise_epsilon (float): Epsilon for noise stability.
            router_temp (float): Temperature for the router logits.
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


    def forward(self, X, router_gate, positional_features, num_positions = None, current_epoch = None):
        """
        Router forward pass with automatic phase management based on epochs.

        Args:
            X (torch.Tensor): Input tensor of shape [N, C, H, W] (or flattened [B*P, ...]).
            router_gate (nn.Module): The gate module to compute logits.
            current_epoch (int, optional): Current training epoch to determine routing phase.
        
        Returns:
            dispatch (torch.Tensor): Dispatch tensor for routing.
            combine (torch.Tensor): Combine tensor for aggregating results.
            z_loss (torch.Tensor): Z-loss value.
            aux_loss (torch.Tensor): Auxiliary load balancing loss.
            logits_std (float): Standard deviation of raw logits (for monitoring).
            logits_temp_std (float): Standard deviation of temperature-scaled logits.
            logits (torch.Tensor): Raw logits (detached, on CPU).
        """

        N = X.shape[0] # Total number of patches
        E = self.num_experts

        # Always compute logits for monitoring and potential use
        logits = router_gate(X, positional_features).to(dtype=torch.float32) # [N, Es]
        logits_std = logits.detach().std().item()

        logits_temp = logits / self.router_temp
        z_loss = self.z_loss(logits_temp)

        logits_temp_std = logits_temp.detach().std().item()
        logits_temp = logits_temp.clamp(min = -10.0, max = 10.0)

        # Route based on current phase (Uniform < 30 epochs, Specialized >= 30 epochs)
        if current_epoch == None:
            return self._specialized_routing(X, logits_temp, logits_std, logits_temp_std, z_loss)
        if current_epoch < 10:
            return self._uniform_routing(X, logits_temp, logits_std, logits_temp_std, z_loss)
        if getattr(router_gate, "is_static_map", False):
            return self._static_expert_region_routing(X, logits_temp, logits_std, logits_temp_std, z_loss, num_positions)
        else:
            return self._specialized_routing(X, logits_temp, logits_std, logits_temp_std, z_loss)
    
    def _static_expert_region_routing(
        self,
        X,
        logits,
        logits_std,
        logits_temp_std,
        z_loss,
        num_positions,
    ):
        """
        Static expert -> region routing.

        X: [B*P, C, H, W]
        logits: [B*P, E], prodotti da StaticFourierMapGate, quindi dipendono solo dalla posizione
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

        probs_pos = F.softmax(logits_pos.float(), dim=-1)  # [P, E]

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
        token_idx = (topk_pos.view(1, k_pos, E) + batch_offsets).reshape(-1)

        expert_idx = (
            torch.arange(E, device=X.device)
            .view(1, 1, E)
            .expand(B, k_pos, E)
            .reshape(-1)
        )

        # Slot locali per esperto: [0 ... B*k_pos-1]
        slot_idx = (
            torch.arange(k_pos, device=X.device)
            .view(1, k_pos, 1)
            .expand(B, k_pos, E)
            + torch.arange(B, device=X.device).view(B, 1, 1) * k_pos
        ).reshape(-1)

        weights = (
            topk_prob.view(1, k_pos, E)
            .expand(B, k_pos, E)
            .reshape(-1)
            .float()
        )

        routing_state = RoutingState(
            token_idx=token_idx,
            expert_idx=expert_idx,
            slot_idx=slot_idx,
            weights=weights,
            num_experts=E,
            num_tokens=N,
            capacity=B * k_pos,
        )

        return routing_state, z_loss, logits_std, logits_temp_std, logits.detach()

    def _specialized_routing(self, X, logits, logits_std, logits_temp_std, z_loss):
        """
        Specialized top-1 routing (Phase 2/3).
        
        Standard Mixture-of-Experts (MoE) routing where tokens are assigned to the 
        expert with the highest probability (Top-1), subject to capacity constraints.
        """
        N = X.shape[0] # Total numbers of patches
        E = self.num_experts
        
        # Adding noise in logits 
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits.float()) * self.noise_std
            logits = logits + noise
        
        probs_e2t = F.softmax(logits.float(), dim = -1) # [N, E]
        # div_loss = self.diverity_loss(probs_e2t)

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

        token_idx = topk_idx.reshape(-1)
        expert_idx = torch.arange(E, device=X.device).unsqueeze(0).expand(k, E).reshape(-1)
        slot_idx = torch.arange(k, device=X.device).unsqueeze(1).expand(k, E).reshape(-1)

        weights = topk_prob.reshape(-1).float()
        overlap_loss = self.compute_overlap_loss(
            probs_e2t = probs_e2t,
            topk_idx = topk_idx, 
            k = k
        )
        balance_loss = self.compute_balance_loss(
            probs_e2t=probs_e2t, 
            topk_idx=topk_idx
        ) 

        routing_state = RoutingState(
            token_idx=token_idx,
            expert_idx=expert_idx, 
            slot_idx=slot_idx,
            weights=weights,
            num_experts=E,
            num_tokens=N,
            capacity=k
        )

        return routing_state, balance_loss, overlap_loss, z_loss, logits_std, logits_temp_std, logits.detach()

    def _uniform_routing(self, X, logits, logits_std, logits_temp_std, z_loss):
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

        routing_state = RoutingState(
            token_idx=token_idx[keep],
            expert_idx=expert_idx[keep],
            slot_idx=slot_idx[keep],
            weights=torch.full(
                (int(keep.sum().item()),),
                1.0,
                device=X.device,
                dtype=torch.float32,
            ),
            num_tokens=N,
            num_experts=E,
            capacity=ccap
        )

        z_loss = torch.tensor(0.0, device=X.device, dtype=torch.float32)
        overlap_loss = torch.tensor(0.0, device=X.device, dtype=torch.float32)
        balance_loss = torch.tensor(0.0, device=X.device, dtype=torch.float32)

        return routing_state, balance_loss, overlap_loss, z_loss, logits_std, logits_temp_std, logits.detach()
    
    def z_loss(self, logits):
        """
        Computes Z-loss to encourage smaller logits.

        Args:
            logits (torch.Tensor): Router logits.
        """

        return torch.logsumexp(logits, dim = -1).square().mean()
    
    def compute_overlap_loss(
        self, 
        probs_e2t, 
        topk_idx, 
        k
    ):
        """
        Pairwise soft overlap loss for expert-choice routing.

        Args:
            probs_e2t: router probabilities after softmax over experts.
                    Shape [N, E]
            topk_idx: unused here, kept for API compatibility.
                    Shape [k, E]
            k: unused here, kept for API compatibility.

        Returns:
            Scalar loss. Lower means less pairwise overlap between experts.
        """
        
        """
        Soft token coverage loss for expert-choice routing.
        Kept under the overlap-loss slot for compatibility with the
        training loop and existing logs.

        Args : 
            probs_e2t : Router probabilities after softmax | Shape [N, E]
            topk_idx : Token indices selected by each expert | Shape [k, E]
            k : Capacity per expert
        """
        N, E = probs_e2t.shape

        if E <= 1 or N <= 1:
            return probs_e2t.new_tensor(0.0)

        q_e2t = probs_e2t / probs_e2t.sum(dim=0, keepdim=True).clamp_min(1e-8)

        expected_assignments = float(k) * q_e2t.sum(dim=1)

        target = probs_e2t.new_tensor(float(E * k) / float(N))

        relative_coverage = expected_assignments / target.clamp_min(1e-8)

        return (relative_coverage - 1.0).pow(2).mean()
        
    def compute_balance_loss(
        self, 
        probs_e2t,
        topk_idx, 
        min_owner_frac = 0.30,
    ) :

        """
        Soft owner balance loss for expert choice 

        Args : 
            probs_e2t : Router probabilities after softmax | Shape [N, E]
            topk_idx : Token indices selected by each expert | Shape [k, E]
            target_entropy : Minimum desired normalized owner entropy 
        """

        N, E = probs_e2t.shape # Shape : [N, E]
        device = probs_e2t.device 
        dtype = probs_e2t.dtype

        if E <= 1 or N <= 1 :
            return probs_e2t.new_tensor(0.0)

        # Build hard assignment matrix S
        S = torch.zeros(
            (N, E),
            device = device, 
            dtype=dtype,
        ) # Shape : [N, E]
        S.scatter_(
            dim = 0, 
            index=topk_idx, 
            src = torch.ones_like(topk_idx, dtype=dtype, device = device)
        ) # Shape : [N, E]

        # Compute soft responsability among assigned experts 
        assigned_prob = probs_e2t * S # Shape : [N, E]

        # Total selected probability mass per token 
        token_mass = assigned_prob.sum(dim = 1, keepdim=True) # Shape : [N, 1]
        processed_mask = (S.sum(dim=1, keepdim=True) > 0).to(dtype) # Shape : [N, 1]

        # For each processed token, sum_e responsability[t, e] = 1
        responsability = assigned_prob / token_mass.clamp_min(1e-8) # Shape : [N, E]
        responsability = responsability * processed_mask # Shape : [N, E]

        # Aggregate soft owner mass per expert 
        owner_mass = responsability.sum(dim = 0) # Shape : [E]

        # Distribution over experts of soft ownership 
        owner_dist = owner_mass / owner_mass.sum().clamp_min(1e-8) # Shape : [E]

        # Penalize if ownership entropy is too low 
        min_owner = probs_e2t.new_tensor(float(min_owner_frac) / float(E))

        # Penalize only experts below the minimum floor
        floor_violation = F.relu(min_owner - owner_dist) / min_owner.clamp_min(1e-8)
        floor_loss = floor_violation.mean()

        return floor_loss
