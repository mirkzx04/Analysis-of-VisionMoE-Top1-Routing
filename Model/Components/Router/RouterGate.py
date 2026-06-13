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

class StaticMapGate(nn.Module):
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
