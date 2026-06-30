import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterGate(nn.Module):
    def __init__(self,
        semantic_channel,
        fourier_channel,
        num_experts,
        patch_h,
        patch_w,
        unified_router,
    ):
        super().__init__()

        self.unified_router = unified_router

        if not unified_router :
            # Initialize router patch representation
            self.semantic_w = nn.Parameter(
                torch.randn(num_experts, semantic_channel, patch_h, patch_w)
            ) # Shape : [E, C_in, H_p, W_p]

            self.position_w = nn.Parameter(
                torch.rand(num_experts, fourier_channel)
            ) # Shape : [E, F]
            # self.position_scale = nn.Parameter(torch.tensor(1.0))
            # self.semantic_scale = nn.Parameter(torch.tensor(1.0))

            self.initialize_weights()
        else :
            self.W = nn.Parameter(
                torch.rand(num_experts, fourier_channel + semantic_channel, patch_h, patch_w)
            )
            nn.init.xavier_uniform_(self.W.flatten(1))

    def forward(self, X, positional_features):
        if self.unified_router :
            Xn = F.layer_norm(X, X.shape[1:]) # [N, C, H, W]

            c_x = Xn.shape[1]
            x_flat = Xn.flatten(1)
            x_w = self.W[:, :c_x].flatten(1).to(dtype=x_flat.dtype)
            logits_sem = x_flat.matmul(x_w.transpose(0, 1)) # [N, E]

            pos_flat = positional_features.flatten(1).to(dtype=logits_sem.dtype)
            pos_w = self.W[:, c_x:].sum(dim=(-1, -2)).to(dtype=pos_flat.dtype)
            logits = logits_sem + pos_flat.matmul(pos_w.transpose(0, 1))

            return logits
        else :
            Xn = F.layer_norm(X, X.shape[1:])

            x_flat = Xn.flatten(1)
            sem_w = self.semantic_w.flatten(1).to(dtype=x_flat.dtype)
            sem_logits = x_flat.matmul(sem_w.transpose(0, 1)) # [N, E]

            pos_flat = positional_features.to(dtype=sem_logits.dtype)
            pos_w = self.position_w.to(dtype=pos_flat.dtype)
            pos_logits = pos_flat.matmul(pos_w.transpose(0, 1)) # [N, E]

        return (sem_logits, pos_logits)

    def initialize_weights(self):
        # semantic_w projects a flattened patch [C, H, W] -> [E], so initialize
        # it like a Linear(C*H*W, E) weight.
        nn.init.xavier_uniform_(self.semantic_w.flatten(1))
        nn.init.xavier_uniform_(self.position_w.flatten(1))

class SemanticOnlyGate(nn.Module):
    """
    Content-only gate (selected by ``semantic_only=True``).

    Routes patches to experts purely by SEMANTIC content: one learnable template
    per expert, ``semantic_w[E, C, H, W]``, matched against the (layer-normalized,
    flattened) patch ``[C, H, W]`` to produce selection logits ``[N, E]``. There is
    NO positional branch -- ``positional_features`` is accepted (Router calls every
    gate as ``gate(X, positional_features)``) but ignored.

    Unlike the decoupled ``RouterGate`` (which returns ``(sem_logits, pos_logits)``
    and SELECTS by position while WEIGHTING by semantics), this gate returns a
    SINGLE logits tensor, so ``Router.forward`` takes the unified single-tensor
    path: top-1 expert-choice in ``_specialized_routing`` with softmax weighting
    over the SAME logits, plus a z-loss on them. No spatial_loss (there is no
    positional map to spread out).

    The weight mirrors the decoupled ``RouterGate.semantic_w`` and is per-layer
    sized: each PCELayer builds its own gate with that layer's ``semantic_channel``
    (= in-channels) and ``patch_h/patch_w`` (= unfold kernel size = patch + 2*halo).
    """
    def __init__(self,
        semantic_channel,
        num_experts,
        patch_h,
        patch_w,
    ):
        super().__init__()

        self.semantic_w = nn.Parameter(
            torch.randn(num_experts, semantic_channel, patch_h, patch_w)
        )  # Shape : [E, C_in, H_p, W_p]

        # semantic_w projects a flattened patch [C, H, W] -> [E], so initialize it
        # like a Linear(C*H*W, E) weight (same convention as RouterGate.semantic_w).
        nn.init.xavier_uniform_(self.semantic_w.flatten(1))

    def forward(self, X, positional_features):
        # positional_features is intentionally unused (content-only routing); it is
        # kept in the signature because Router calls every gate as gate(X, pos_feat).
        Xn = F.layer_norm(X, X.shape[1:])  # [N, C, H, W]

        x_flat = Xn.flatten(1)
        sem_w = self.semantic_w.flatten(1).to(dtype=x_flat.dtype)
        logits = x_flat.matmul(sem_w.transpose(0, 1))  # [N, E]

        return logits

class PosOnlyGate(nn.Module):
    """
    Position-only gate (selected by ``pos_only=True``).

    Learnable per-position logits over experts (``pos_coeff[E, P]``); routing is
    done downstream by ``Router._static_expert_region_routing`` (softmax + expert
    choice top-k). This is the *former* ``StaticMapGate``; when the LitModule sets
    ``static="not_learnable"`` it freezes ``pos_coeff`` at its random init, i.e. a
    random (but frozen) position->expert preference map.
    """
    def __init__(self, num_experts, P):
        super().__init__()
        self.num_positions = P
        self.num_experts = num_experts

        self.pos_coeff = nn.Parameter(torch.empty(num_experts, P))
        nn.init.xavier_uniform_(self.pos_coeff)

        self.is_static_map = True

    def forward(self, X, positional_features):
        N = X.shape[0]
        P = self.num_positions
        E = self.num_experts

        if N % P != 0:
            raise ValueError(
                f"N={N} must be divisible by P={P}. "
                f"Static gate num_positions is wrong for this layer."
            )

        B = N// P

        # [E, P] -> [P, E]
        logit_pos = self.pos_coeff.T
        logits = logit_pos.unsqueeze(0).expand(B, P, E).reshape(N, E)

        return logits.float()


class StaticMapGate(nn.Module):
    """
    Deterministic, parameter-free position -> expert map (selected by
    ``use_static_map=True``).

    Each of the ``P`` positions of the (square) ``H x W`` patch grid is assigned to
    EXACTLY ONE expert by nearest-center / Voronoi over a near-square lattice of
    ``E`` centers in ``[0, 1]^2`` (e.g. ``E=16`` -> 4x4 lattice). Experts therefore
    own disjoint, spatially compact groups of positions. The Voronoi rule handles a
    grid not divisible by ``E`` (e.g. 7x7 / 16) without edge cases.

    There is NO softmax, NO capacity / top-k, NO learnable parameter: routing weight
    is fixed to 1.0. The fixed map is precomputed once in ``__init__`` and stored as
    (non-persistent) buffers; ``build_routing_state`` materializes the exact partition
    dispatch for a given batch, and ``forward`` returns one-hot logits used only for
    monitoring / offline analysis.
    """
    def __init__(self, num_experts, P):
        super().__init__()
        self.num_experts = num_experts
        self.num_positions = P

        # is_static_map keeps the LitModule freeze logic and the static logit-stats
        # logging working; is_deterministic_map triggers the dedicated routing branch
        # in Router.forward.
        self.is_static_map = True
        self.is_deterministic_map = True

        H = math.isqrt(P)
        if H * H != P:
            raise ValueError(
                f"StaticMapGate expects a square patch grid, got P={P} (isqrt={H}). "
                f"Thread explicit h/w to the gate for a non-square layer."
            )
        W = H

        pos2expert = self._build_pos2expert(P, num_experts, H, W)          # [P] long
        pos_table, valid_mask = self._build_dispatch_tables(pos2expert, num_experts)

        # Fully determined by (P, E) -> recomputed on every construction, so keep them
        # out of the checkpoint (persistent=False).
        self.register_buffer("pos2expert", pos2expert, persistent=False)
        self.register_buffer("pos_table", pos_table, persistent=False)      # [E, K_per]
        self.register_buffer("valid_mask", valid_mask, persistent=False)    # [E, K_per]

    @staticmethod
    def _expert_centers(E):
        """Near-square lattice of ``E`` cell-centered points in ``[0, 1]^2``."""
        g_rows = max(1, int(round(math.sqrt(E))))
        base, rem = divmod(E, g_rows)
        centers = []
        for r in range(g_rows):
            cols_r = base + (1 if r < rem else 0)
            cy = (r + 0.5) / g_rows
            for j in range(cols_r):
                cx = (j + 0.5) / cols_r
                centers.append((cy, cx))
        return torch.tensor(centers, dtype=torch.float32)                   # [E, 2]

    @classmethod
    def _build_pos2expert(cls, P, E, H, W):
        idx = torch.arange(P)
        pos_y = (idx // W).to(torch.float32).add(0.5).div(H)
        pos_x = (idx % W).to(torch.float32).add(0.5).div(W)
        pos_2d = torch.stack([pos_y, pos_x], dim=1)                         # [P, 2]
        centers = cls._expert_centers(E)                                    # [E, 2]
        dist = torch.cdist(pos_2d, centers)                                 # [P, E]
        return dist.argmin(dim=1).to(torch.long)                           # [P]

    @staticmethod
    def _build_dispatch_tables(pos2expert, E):
        """Per-expert owned-position table padded to a rectangular [E, K_per]."""
        owned = [(pos2expert == e).nonzero(as_tuple=False).flatten() for e in range(E)]
        counts = [int(o.numel()) for o in owned]
        K_per = max(1, max(counts))
        pos_table = torch.zeros(E, K_per, dtype=torch.long)
        valid_mask = torch.zeros(E, K_per, dtype=torch.float32)
        for e, o in enumerate(owned):
            c = counts[e]
            if c > 0:
                pos_table[e, :c] = o
                valid_mask[e, :c] = 1.0
        return pos_table, valid_mask

    def forward(self, X, positional_features):
        # One-hot logits, used ONLY for monitoring / analysis (argmax recovers the
        # deterministic assignment). No softmax here; the actual routing is built by
        # build_routing_state().
        N = X.shape[0]
        B = N // self.num_positions
        e_idx = self.pos2expert.to(X.device).repeat(B)                     # [N]
        return F.one_hot(e_idx, self.num_experts).to(torch.float32)        # [N, E]

    def build_routing_state(self, N, device):
        """
        Materialize the exact deterministic partition as a RoutingState.

        Position p -> expert pos2expert[p] with weight 1.0, replicated over the B
        images. Padding slots (rectangular [E, K_per] table) carry weight 0.0 and a
        sentinel token, so they add nothing in the aggregation (same convention as
        Router._uniform_routing).
        """
        # Local import avoids any import-time coupling with Router (no cycle).
        from Model.Components.Router.Router import RoutingState

        P, E = self.num_positions, self.num_experts
        if N % P != 0:
            raise ValueError(f"N={N} must be divisible by P={P}.")
        B = N // P
        K_per = self.pos_table.shape[1]
        K = B * K_per

        pos_table = self.pos_table.to(device)                              # [E, K_per]
        valid = self.valid_mask.to(device)                                 # [E, K_per]

        batch_off = (torch.arange(B, device=device) * P).view(1, B, 1)     # [1, B, 1]
        token = (pos_table.view(E, 1, K_per) + batch_off).reshape(E, K)    # [E, K]
        weights = valid.view(E, 1, K_per).expand(E, B, K_per).reshape(E, K)

        expert_idx = torch.arange(E, device=device).view(E, 1).expand(E, K)
        slot_idx = torch.arange(K, device=device).view(1, K).expand(E, K)

        return RoutingState(
            token_idx=token.reshape(-1),
            expert_idx=expert_idx.reshape(-1),
            slot_idx=slot_idx.reshape(-1),
            weights=weights.reshape(-1),
            num_experts=E,
            num_tokens=N,
            capacity=K,
        )

class InteractionGate(nn.Module):
    def __init__(
        self,
        num_exp,
        sem_c,
        f_c,
        patch_h,
        patch_w,
        rank = 4,
    ):
        super().__init__()

        self.rank = rank
        self.num_exp = num_exp

        self.sem_proj = nn.Parameter(
            torch.empty(
                num_exp,
                rank,
                sem_c,
                patch_h,
                patch_w,
            )
        )

        self.pos_proj = nn.Parameter(torch.empty(num_exp, rank, f_c))
        self.bias = nn.Parameter(torch.zeros(num_exp))
        # self.lam = nn.Parameter(torch.tensor(1.0))

        self.reset_parameters()

    def reset_parameters(self):
        # Inizializza ogni coppia expert-component come una proiezione D -> 1.
        sem_flat = self.sem_proj.view( self.num_exp * self.rank, -1)
        pos_flat = self.pos_proj.view(self.num_exp * self.rank,-1)

        nn.init.xavier_uniform_(sem_flat)
        nn.init.xavier_uniform_(pos_flat)
        nn.init.zeros_(self.bias)
    def forward(self, x, positional_features):
        x = F.layer_norm(x.float(),x.shape[1:]).flatten(1)

        pos = positional_features.float().flatten(1)
        pos = F.layer_norm(pos, (pos.shape[-1],))

        sem_score = torch.einsum(
            "nd,erd->ner",
            x,
            self.sem_proj.flatten(2),
        ) 

        pos_score = torch.einsum(
            "nf,erf->ner",
            pos,
            self.pos_proj,
        ) 

        joint_score = (sem_score  * pos_score)
        logits =  (joint_score.sum(dim = -1) / math.sqrt(self.rank)) + self.bias
        return (logits).float()


        