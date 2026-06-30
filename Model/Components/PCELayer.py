import torch

from torch import nn

from .ConvExpert import ConvExpert
from Model.Components.Router.RouterGate import RouterGate, StaticMapGate, PosOnlyGate, SemanticOnlyGate, InteractionGate

class PCELayer(nn.Module):
    def __init__(self,
                inpt_channel,
                out_channel,
                num_experts,
                patch_size,
                fourie_freq,
                gate_channel,
                kernel_size,
                unfold_kernel_size,
                num_positions = None,
                use_static_map = False,
                pos_only = False,
                semantic_only = False,
                unified_router = False,
                interaction = False,
                interaction_hidden_size = 128,
                interaction_include_main_effects = False,
                ):
        super().__init__()
        self.experts = nn.ModuleList([
            ConvExpert(
                in_channel=inpt_channel,
                out_channel=out_channel,
                use_residual=True,
                kernel_size = kernel_size
            )
            for _ in range(num_experts)
        ])
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        if use_static_map :
            # Deterministic fixed position->expert Voronoi map (no params, weight 1.0).
            self.router_gate = StaticMapGate(
                num_experts=num_experts,
                P = num_positions,
            )
        elif pos_only :
            # Learnable position-only gate (former StaticMapGate behaviour).
            self.router_gate = PosOnlyGate(
                num_experts=num_experts,
                P = num_positions,
            )
        elif semantic_only :
            # Content-only gate: selection by semantics, no positional branch.
            self.router_gate = SemanticOnlyGate(
                semantic_channel=inpt_channel,
                num_experts=num_experts,
                patch_h=unfold_kernel_size,
                patch_w=unfold_kernel_size,
            )
        elif interaction :
            # Interaction gate: semantic x positional feature interaction -> single
            # logits tensor [N, E]; routed like the content-only gate (top-1 select +
            # softmax weighting on the same logits) via Router._specialized_routing.
            self.router_gate = InteractionGate(
                num_exp=num_experts,
                sem_c=inpt_channel,
                f_c=2 + 4 * fourie_freq,
                patch_h=unfold_kernel_size,
                patch_w=unfold_kernel_size,
            )
        else :
            self.router_gate = RouterGate(
                semantic_channel=inpt_channel,
                fourier_channel= 2 + 4 * fourie_freq,
                num_experts=num_experts,
                patch_h=unfold_kernel_size,
                patch_w=unfold_kernel_size,
                unified_router = unified_router,
            )
                
        self.patch_size = patch_size
        self.fourier_freq = fourie_freq
        self.unfold_kernel_size = unfold_kernel_size        

        self.post_block = nn.Sequential(nn.Conv2d(
            out_channels=out_channel, 
            in_channels=out_channel, 
            groups=out_channel, 
            kernel_size=3, 
            padding=1, 
            bias=False
        ))
        nn.init.zeros_(self.post_block[0].weight)
