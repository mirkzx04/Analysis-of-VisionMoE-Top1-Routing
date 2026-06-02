from numpy import dtype, indices
import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterGate(nn.Module):
    def __init__(self, semantic_channel, fourier_channel, num_experts, patch_h, patch_w):
        super().__init__()
        
        # Initialize router patch representation 
        self.semantic_w = nn.Parameter(
            torch.randn(num_experts, semantic_channel, patch_h, patch_w)
        ) # Shape : [E, C_in, H_p, W_p]

        self.position_w = nn.Parameter(
            torch.rand(num_experts, fourier_channel)
        ) # Shape : [E, F]

        self.initialize_weights()

    def forward(self, X, positional_features):
        Xn = F.layer_norm(X, X.shape[1:])

        sem_logits = torch.einsum("nchw, echw -> ne", Xn, self.semantic_w)
        pos_logits = torch.einsum("nf, ef -> ne", positional_features.flatten(1), self.position_w)

        return (sem_logits + pos_logits).float()

    def initialize_weights(self):
        # expert_emb is used as a linear projection from a flattened patch
        # [C, H, W] -> [E], so initialize it like a Linear(C*H*W, E) weight.
        nn.init.xavier_uniform_(self.semantic_w.flatten(1))
        nn.init.xavier_uniform_(self.position_w.flatten(1))

class StaticFourierMapGate(nn.Module):
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