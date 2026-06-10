import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from Model.Components.Router import Router
from Model.Components.PCELayer import PCELayer
from Model.Components.DownsampleResBlock import DownsampleResBlock

from Model.Components.MoEAggregator import MoEAggregator
from Datasets_Classes.PatchExtractor import PatchExtractor
from Model.Components.Router.Router import Router
from Testing.dataclass.DebuggerDataClass import RoutingDebug

class PCENetwork(nn.Module):
    def __init__(self,
                    num_experts,
                    layer_number,
                    patch_size,
                    num_classes,
                    router_temp,
                    capacity_factor_train,
                    capacity_factor_val,
                    halo_for_patches,
                    input_size = 224,
                    use_static_map = False,
                    unified_router = False,
                    task = "seg",
                    uniform_epochs = 10
                 ):
        super().__init__()

        """
        Constructor of PCE Network

        Args :
            kernel_sz_exps (int) -> kernel size of experts
            out_cha_exps // -> out channel for convolution in experts
            num_experts // -> Number of experts per layer
            out_cha_router // -> out channel for conv projection in router
            layer_number // -> Numer of layers
            dropout (float) -> dropout probability of the convolution experts
            patch_size (int) -> Size of patches, used in PatchExtractor
            router (Object.router) -> Router object, used for routing through experts
            threshold (float) -> Threshold for experts scores, used in router
        """

        self.num_classes = num_classes
        self.num_experts = num_experts
        self.halo_for_patches = halo_for_patches
        self.input_size = input_size
        self.use_static_map = use_static_map
        self.unified_router = unified_router
        self.task = task

        # Mixed-precision inference
        self.amp_inference = True
        self.amp_dtype = torch.bfloat16

        self.patch_extractor = PatchExtractor(patch_size)
        self.layers = nn.ModuleList()
        last_channel = self.create_layers(
            num_experts=num_experts,
            layer_number=layer_number,
            input_size = input_size,
            use_static_map = use_static_map,
            unified_router = unified_router
        )

        self.router = Router(
            num_experts=num_experts,
            num_layers=layer_number,
            router_temp = router_temp,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_val,
            uniform_epochs=uniform_epochs,
        )

        if task == "class" : 
            self.pooler = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.prediction_head = nn.Sequential(
                nn.Linear(last_channel, num_classes),
            )
        if task == "seg" :
            self.prediction_head = nn.Sequential(
                nn.Conv2d(last_channel, last_channel // 2, kernel_size= 3 , padding= 1, bias=False),
                nn.BatchNorm2d(last_channel // 2),
                nn.SiLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(last_channel // 2, last_channel // 4, kernel_size= 3, padding= 1, bias= False),
                nn.BatchNorm2d(last_channel // 4),
                nn.SiLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(last_channel // 4, num_classes, kernel_size=1)
            )
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        num_experts_per_layer = [
            len(l.experts) if isinstance(l, PCELayer) else 0
            for l in self.layers
        ]
        self.moe_aggregator = MoEAggregator(
            num_layers=len(num_experts_per_layer),
            num_experts=num_experts_per_layer,
        )
    def create_layers(self, num_experts, layer_number, input_size, use_static_map, unified_router):
        """
        Create layers of PCE Network

        Args:
            num_experts : Number of experts for layer l
            layer_number : Number of ResNet Layer
            input_size : Dimension of the input H x W
            use_static_map : Set if the model use the static or dynamic routing mapping
        """
        def get_fourie_channel(F):
            return 2 + 4 * F

        def conv_out_size(size, kernel_size, stride, padding, dilation = 1):
            return ((size + 2 * padding - dilation * (kernel_size - 1) -  1) // stride) + 1

        # Setting start parameters
        patch_size = self.patch_extractor.patch_size
        unfold_kernel_size = patch_size + 2 * self.halo_for_patches

        fourier_freq = self.patch_extractor.num_frequencies
        fourier_channel =  get_fourie_channel(fourier_freq)
        inpt_channel = 64
        out_channel = 64

        # After stem : Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)
        current_h = conv_out_size(input_size, kernel_size=7, stride=2, padding=3)
        current_w = conv_out_size(input_size, kernel_size=7, stride=2, padding=3)

        for l in range(layer_number):
            current_gate_channel = inpt_channel + fourier_channel

            if l in [2, 4, 6]:
                transition_out = inpt_channel * 2
                self.layers.append(
                    DownsampleResBlock(in_ch = inpt_channel, out_ch= transition_out)
                )

                inpt_channel = transition_out
                out_channel = transition_out

                patch_size = max(2, patch_size // 2)
                unfold_kernel_size = patch_size + 2 * self.halo_for_patches

                # After downsample
                current_h = conv_out_size(current_h, kernel_size=3, stride=2, padding=1)
                current_w = conv_out_size(current_w, kernel_size=3, stride=2, padding=1)

            else:
                # Compute number of patches
                h_position = current_h // patch_size
                w_position = current_w // patch_size
                P = h_position * w_position

                self.layers.append(PCELayer(
                    inpt_channel=inpt_channel,
                    out_channel=out_channel,
                    num_experts=num_experts,
                    patch_size=patch_size,
                    fourie_freq=fourier_freq,
                    gate_channel=current_gate_channel,
                    kernel_size = 3,
                    unfold_kernel_size = unfold_kernel_size,
                    num_positions = P,
                    use_static_map = use_static_map,
                    unified_router = unified_router,
                ))

        return out_channel

    def crop_center(self, y, patch_size):
        h = self.halo_for_patches
        if h == 0:
            return y
        return y[:, :, h:h + patch_size, h:h + patch_size]

    def forward(self, X, current_epoch=None, collect_routes = False, collect_debug = None):
        """
        Forward method of PCE Network

        Args :
            X (torch.tensor) -> input of network, tensor.shape = (B,C,H,W)
            current_epoch (int) -> current training epoch, used to determine routing strategy

        Pipeline of PCE Network:
            1. Divide input img/feature map in patches
            2. For each layer:
                2.1. Extract patches from input img/feature map
                2.2. Get experts scores from router
                2.3. For each expert:
                    2.3.1. Apply expert to patches
                    2.3.2. Concatenate experts outputs with weighted sum
                2.4. Reassemble patches in a single image
                2.5. Apply final convolution 1x1 to the output of the layer
            3. Create linear layer for classification if not exists
            4. Return logits and expert scores

        Returns:
            logits (torch.tensor) : tensor beatches (B, num_classes)
        """
        tot_z_loss  = 0.0
        tot_spatial_loss = 0.0

        img_device = X.device
        B = X.shape[0]
        in_hw = X.shape[-2:]
        routing_debug = []

        # collect_debug controls only the (expensive) host-side routing dump used by
        # offline analysis
        if collect_debug is None:
            collect_debug = collect_routes

        moe_layers = 0

        # bf16 autocast for the convolutional path only
        amp_enabled = self.amp_inference and (not self.training) and X.is_cuda

        def amp_ctx():
            if amp_enabled:
                return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            return contextlib.nullcontext()

        with amp_ctx():
            X = self.stem(X)
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, DownsampleResBlock):
                with amp_ctx():
                    X = layer(X)
            else :
                moe_layers += 1

                # Store layer attributes for cleaner access
                experts = layer.experts
                merge_gn = layer.merge_gn
                merge_act = layer.activation_merge
                post_block = layer.post_block
                unfold_kernel_size = layer.unfold_kernel_size

                # Extract patches from the current feature map
                patch_size = layer.patch_size
                self.patch_extractor.patch_size = patch_size

                # Get token from X patches
                h_patches, w_patches, X_patches = self.patch_extractor.get_patches(
                    image=X,
                    unfold_kernel_size = unfold_kernel_size,
                    halo = self.halo_for_patches
                )
                P, C, H, W = X_patches.shape[1:] # Shape : [B, P, C, H, W]
                N = B*P

                X_tokens = X_patches.reshape(B * P, C, H, W)

                # Get positional features from fourier
                positional_features = self.patch_extractor.get_positional(
                    h_patches=h_patches,
                    w_patches=w_patches,
                    B=B,
                    img_device=img_device,
                ) # Shape : [B, P, FC]
                positional_features = positional_features.flatten(0, 1)

                # Route patches to experts and compute auxiliary losses.
                with torch.autocast(device_type="cuda", enabled=False):
                    routing_state, spatial_loss, z_loss, logits, logit_stats = self.router(
                        X_tokens.float(),
                        layer.router_gate,
                        positional_features.float(),
                        num_positions=P,
                        h_patches=h_patches,
                        w_patches=w_patches,
                        current_epoch = current_epoch,
                        collect_metrics=collect_routes,
                    )

                if collect_routes:
                    valid_route = routing_state.weights > 0
                    if collect_debug:
                        with torch.no_grad():
                            _gate_out = layer.router_gate(X_tokens.float(), positional_features.float())
                            _sem_logits = _gate_out[0].detach().cpu() if isinstance(_gate_out, tuple) else None
                            _patch_var = X_tokens.detach().float().var(dim=[1, 2, 3]).cpu()
                        routing_debug.append(
                            RoutingDebug(
                                layer_idx=layer_idx,
                                B = B,
                                E = len(experts),
                                h_patches=h_patches,
                                w_patches=w_patches,
                                token_idx=routing_state.token_idx[valid_route].detach().cpu(),
                                experts_idx=routing_state.expert_idx[valid_route].detach().cpu(),
                                weight = routing_state.weights[valid_route].detach().cpu(),
                                logits = logits.detach().cpu(),
                                sem_logits = _sem_logits,
                                patch_var = _patch_var,
                            )
                        )

                E = int(routing_state.num_experts)
                K = int(routing_state.capacity)
                dispatch_token_idx = routing_state.token_idx.view(E, K)
                dispatch_weights = routing_state.weights.view(E, K).to(dtype=X_tokens.dtype)
                X_center = self.crop_center(X_tokens, patch_size)
                outputs = X_center.clone()
                rel_delta_sum = None
                rel_delta_count = None
                rel_delta_eps = 1e-8

                for e, expert in enumerate(experts):
                    n_e = dispatch_token_idx[e]
                    x_e = X_tokens.index_select(0, n_e)
                    x_e_center = self.crop_center(x_e, patch_size)

                    with amp_ctx():
                        y_e = expert(x_e)
                    y_e = self.crop_center(y_e, patch_size)

                    delta_e = y_e - x_e_center
                    w_e = dispatch_weights[e].view(-1, 1, 1, 1)
                    outputs.index_add_(0, n_e, (delta_e * w_e).to(dtype=outputs.dtype))

                    if collect_routes:
                        with torch.no_grad():
                            valid_e = dispatch_weights[e] > 0
                            rel_delta = delta_e.detach().flatten(1).norm(dim=1) / (
                                x_e_center.detach().flatten(1).norm(dim=1) + rel_delta_eps
                            )
                            # Keep the reduction on-device (mask instead of boolean
                            # index + .item()) so no GPU->CPU sync happens per expert;
                            # masked-out entries are exact zeros and change neither
                            # the sum nor the count.
                            rel_delta = torch.where(valid_e, rel_delta, torch.zeros_like(rel_delta))
                            batch_rel_delta_sum = rel_delta.sum()
                            batch_rel_delta_count = valid_e.sum()
                            rel_delta_sum = (
                                batch_rel_delta_sum if rel_delta_sum is None
                                else rel_delta_sum + batch_rel_delta_sum
                            )
                            rel_delta_count = (
                                batch_rel_delta_count if rel_delta_count is None
                                else rel_delta_count + batch_rel_delta_count
                            )

                # Reassemble patches back into spatial feature map
                _, C_out, H_out, W_out = outputs.shape

                if collect_routes :
                    with torch.no_grad():
                        eps = 1e-8

                        expert_effective_delta = outputs.detach() - X_center.detach()

                        expert_effective_rel_delta = (
                            expert_effective_delta.flatten(1).norm(dim=1) / (X_center.detach().flatten(1).norm(dim=1) + eps)
                        )

                        expert_effective_rel_delta_sum = expert_effective_rel_delta.sum()
                        expert_effective_rel_delta_count = int(expert_effective_rel_delta.numel())

                    with torch.no_grad():
                        self.moe_aggregator.update_layer(
                            layer_idx=layer_idx,
                            token_idx=routing_state.token_idx[valid_route].detach(),
                            num_tokens=routing_state.num_tokens,
                            logit_stats=logit_stats,
                            logits=logits.detach(),
                            expert_idx=routing_state.expert_idx[valid_route].detach(),
                            weights=routing_state.weights[valid_route].detach(),
                            rel_delta_sum=rel_delta_sum,
                            rel_delta_count=rel_delta_count,
                            expert_effective_rel_delta_sum=expert_effective_rel_delta_sum,
                            expert_effective_rel_delta_count=expert_effective_rel_delta_count,
                            num_positions=P,
                        )

                X = rearrange(
                    outputs.view(B, P, C_out, H_out, W_out),
                    'b (h w) c ph pw -> b c (h ph) (w pw)',
                    h=h_patches, w=w_patches
                )
                X_sparse = X

                # Dense and residual block
                with amp_ctx():
                    dens_in = merge_act(merge_gn(X))
                    dense_delta = post_block(dens_in)
                X = X + dense_delta

                if collect_routes :
                    with torch.no_grad():
                        eps = 1e-8

                        sparse_norm = X_sparse.detach().flatten(1).norm(dim=1).mean()
                        dense_rel_delta = (
                            dense_delta.detach().flatten(1).norm(dim = 1).mean() / (sparse_norm + eps)
                        )

                        self.moe_aggregator.update_dense_layer(
                            layer_idx=layer_idx,
                            dense_rel_delta=dense_rel_delta,
                            device=X.device,
                        )

                tot_z_loss += z_loss
                tot_spatial_loss += spatial_loss

        # Apply global pooling and prediction head
        with amp_ctx():
            if self.task == "class" :
                x = self.pooler(X)
                x = self.flatten(x)
                logits = self.prediction_head(x)
            elif self.task == "seg" :
                logits = self.prediction_head(X)
                logits = F.interpolate(logits, size = in_hw, mode="bilinear", align_corners = False)

        logits = logits.float()

        tot_z_loss = tot_z_loss / moe_layers
        tot_spatial_loss = tot_spatial_loss / moe_layers

        if collect_routes:
            return logits, tot_spatial_loss, tot_z_loss, routing_debug

        return logits, tot_spatial_loss, tot_z_loss,


    def _indices_from_dispatch(self, dispatch):
        """
        Get indices from dispatch

        Args:
            dispatch (torch.tensor) : tensor of shape [N, E, Ccap]
        """
        idx = dispatch.nonzero(as_tuple = False)
        device = dispatch.device
        return idx[:, 0], idx[:, 1], idx[:, 2]

    @torch.no_grad
    def _router_metrics(self, dispatch, combine, N, E, Ccap):
        """
        Get router metrics

        Args:
            dispatch (torch.tensor) : tensor of shape [N, E, Ccap]
            combine (torch.tensor) : tensor of shape [N, E, Ccap]
        """
        # Get assign patches
        assign_patches = dispatch.view(N, -1).sum(dim=1)
        dropped_mask = (assign_patches == 0)
        dropped_patches = dropped_mask.float().mean().item() # Percentage of dropped patches

        # Get processed patches
        processed_patches = int((assign_patches > 0).sum().item()) # Number of processed patches
        capacity_total = E * Ccap
        capacity_efficiency = (processed_patches / max(1, capacity_total)) # Percentage of capacity used

        # Get usage of experts
        patches_to_exp = dispatch.any(dim=2) # [N, E]
        usage_counts = patches_to_exp.sum(dim=0) # [E] Count of patches assigned to each expert
        usage_frac = (usage_counts.float() / max(1, N)) # [E] Percentage of patches assigned to each expert

        dead_experts = (int(usage_counts == 0).sum().item()) # Number of dead experts
        alive_experts = (int(usage_counts > 0).sum().item()) # Number of alive experts
        min_usage = float(usage_frac.min().item()) # Minimum usage of experts
        max_usage = float(usage_frac.max().item()) # Maximum usage of experts
        mean_usage = float(usage_frac.mean().item()) # Mean usage of experts

        imbalance_index = (max_usage / max(min_usage, 1e-12)) if E > 1 else 0.0 # Compute imbalance index

        # Compute entropy and normalize it
        if E > 1:
            p = usage_frac + 1e-12
            p = p / p.sum() # Normalize
            entropy = float((-(p * p.log()).sum().item())) # Compute shannon entropy
            entropy_norm = entropy / float(torch.log(torch.tensor(float(E))).item())

        # Compute mean and std of usage of experts
        mean_u = float(usage_counts.float().mean().item()) if E > 1 else 0.0 # Mean number of patches assigned to each expert
        std_u = float(usage_counts.float().std().item()) if E > 1 else 0.0 # Standard deviation of number of patches assigned to each expert
        cov_usage = float(std_u / max(1e-12, mean_u)) if E > 1 else 0.0 # Covariance of number of patches assigned to each expert

        # Compute capacity usage of experts
        slot_used = dispatch.any(dim = 0) # [E, Ccap]
        capacity_used = slot_used.sum(dim = 1).float() # number of slots used by each expert [E]
        capacity_ratio_per_exp = (capacity_used / max(1, Ccap)) # percentage of capacity used by each expert [E]
        mean_capacity_ratio = float(capacity_ratio_per_exp.mean().item()) # mean percentage of capacity used by each expert
        p95_capacity_ratio = float(capacity_ratio_per_exp.quantile(0.95).item()) # 95th percentile of percentage of capacity used by each expert

        # Collision (Saity check : 0.0)
        slot_counts = dispatch.sum(dim = 0) # [E, Ccap]
        collisions = int((slot_counts > 1).sum().item()) # number of collisions
        max_slot_count = int(slot_counts.max().item()) if slot_counts.numel() else 0 # maximum number of slots used by an expert
        if collisions >= 0:
            print(f'=== ALERT : COLLISIONS : {collisions} ===')

        # Gate per patches
        gate_per_patches = combine.view(N, -1).sum(dim = 1) # [N]
        mean_gate = float(gate_per_patches.mean().item()) # mean gate per patch
        p10_gate = float(gate_per_patches.quantile(0.10).item()) # 10th percentile of gate per patch
        p50_gate = float(gate_per_patches.quantile(0.50).item()) # 50th percentile of gate per patch
        p90_gate = float(gate_per_patches.quantile(0.90).item()) # 90th percentile of gate per patch

        # Gate per expert
        gate_per_exp_sum = combine.sum(dim = (0, 2)) # [E]
        counts = usage_counts.clamp_min(1)
        avg_gate_per_exp = float(gate_per_exp_sum / counts).float()
        mean_avg_gate_per_exp = float(avg_gate_per_exp.mean().item()) if E > 1 else 0.0
        min_avg_gate_per_exp = float(avg_gate_per_exp.min().item()) if E > 1 else 0.0 # Low : router has no confidence in expert
        max_avg_gate_per_exp = float(avg_gate_per_exp.max().item()) if E > 1 else 0.0 # High : router has confidence in expert
