import torch

from torch import nn

from .ConvExpert import ConvExpert
from Model.Components.Router.RouterGate import RouterGate, StaticMapGate

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
                unified_router = False,
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
        self.merge_gn = nn.GroupNorm(num_groups=min(8, out_channel), num_channels=out_channel)
        self.activation_merge = nn.SiLU(inplace=True)

        if use_static_map : 
            self.router_gate = StaticMapGate(
                num_experts=num_experts,
                P = num_positions,
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

        self.post_block = nn.Sequential(
            nn.Conv2d(out_channels=out_channel, in_channels=out_channel, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_channels=out_channel, num_groups=min(8, out_channel)),
            nn.SiLU(inplace=True)
        )
        nn.init.zeros_(self.post_block[0].weight)
