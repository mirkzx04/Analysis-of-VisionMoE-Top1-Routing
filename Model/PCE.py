import torch
import torch.nn as nn

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

        self.patch_extractor = PatchExtractor(patch_size)
        self.layers = nn.ModuleList()
        last_channel = self.create_layers(
            num_experts=num_experts,
            layer_number=layer_number,
            input_size = input_size, 
            use_static_map = use_static_map
        )        

        self.router = Router(
            num_experts=num_experts,
            num_layers=layer_number,
            router_temp = router_temp,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_val,
        )

        self.pooler = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.prediction_head = nn.Sequential(
            nn.Linear(last_channel, num_classes),
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
    def create_layers(self, num_experts, layer_number, input_size, use_static_map):
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
                    use_static_map = use_static_map
                ))
                
        return out_channel

    def crop_center(self, y, patch_size):
        h = self.halo_for_patches
        if h == 0:
            return y
        return y[:, :, h:h + patch_size, h:h + patch_size]

    def forward(self, X, current_epoch=None, collect_routes = False):
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
        tot_balance_loss = 0.0
        tot_overlap_loss = 0.0

        img_device = X.device
        B = X.shape[0]
        routing_debug = []

        moe_layers = 0

        X = self.stem(X)
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, DownsampleResBlock):
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
                )
                positional_features = positional_features.flatten(0, 1)
                
                # Route patches to experts and compute auxiliary losses
                routing_state, balance_loss, overlap_loss, z_loss, logits_std, logits_temp_std, logits = self.router(
                    X_tokens, 
                    layer.router_gate, 
                    positional_features,
                    num_positions=P,
                    current_epoch = current_epoch,
                )

                if collect_routes:
                    routing_debug.append(
                        RoutingDebug(
                            layer_idx=layer_idx, 
                            B = B,
                            E = len(experts), 
                            h_patches=h_patches,
                            w_patches=w_patches,
                            token_idx=routing_state.token_idx.detach().cpu(),
                            experts_idx=routing_state.expert_idx.detach().cpu(),
                            weight = routing_state.weights.detach().cpu(),
                            logits = logits.detach().cpu()
                        )
                    )

                token_idx = routing_state.token_idx
                expert_idx = routing_state.expert_idx
                weights = routing_state.weights.to(dtype = X_tokens.dtype)

                order = torch.argsort(expert_idx)
                token_idx = token_idx.index_select(0, order)
                expert_idx = expert_idx.index_select(0, order)
                weights = weights.index_select(0, order)

                E = len(experts)
                counts = torch.bincount(expert_idx, minlength=E)

                per_exp_token_idx = [None] * E
                per_exp_weights = [None] * E
                offset = 0
                for e, count in enumerate(counts.tolist()):
                    if count == 0:
                        continue
                    next_offset = offset + count
                    per_exp_token_idx[e] = token_idx[offset:next_offset]
                    per_exp_weights[e] = weights[offset:next_offset].view(-1, 1, 1, 1)
                    offset = next_offset
                
                X_center = self.crop_center(X_tokens, patch_size)
                outputs = X_center.clone()
                rel_delta_sum = 0.0
                rel_delta_count = 0
                rel_delta_eps = 1e-8

                for e, expert in enumerate(experts):
                    n_e = per_exp_token_idx[e]
                    if n_e is None:
                        continue

                    x_e = X_tokens.index_select(0, n_e)
                    x_e_center = self.crop_center(x_e, patch_size)

                    y_e = expert(x_e)
                    y_e = self.crop_center(y_e, patch_size)

                    w_e = per_exp_weights[e] # Shape : [k_e, 1, 1, 1]

                    delta_e = y_e - x_e_center
                    with torch.no_grad():
                        rel_delta = delta_e.detach().flatten(1).norm(dim=1) / (
                            x_e_center.detach().flatten(1).norm(dim=1) + rel_delta_eps
                        )
                        rel_delta_sum += float(rel_delta.sum().item())
                        rel_delta_count += int(rel_delta.numel())
                    contrib = delta_e * w_e.to(dtype=y_e.dtype)
                    outputs.index_add_(0, n_e, contrib)

                # Reassemble patches back into spatial feature map
                _, C_out, H_out, W_out = outputs.shape

                with torch.no_grad():
                    eps = 1e-8

                    expert_effective_delta = outputs.detach() - X_center.detach()

                    expert_effective_rel_delta = (
                        expert_effective_delta.flatten(1).norm(dim=1) / (X_center.detach().flatten(1).norm(dim=1) + eps)
                    )

                    expert_effective_rel_delta_sum = float(expert_effective_rel_delta.sum().item())
                    expert_effective_rel_delta_count = int(expert_effective_rel_delta.numel())

                with torch.no_grad():
                    self.moe_aggregator.update_layer(
                        layer_idx=layer_idx,
                        token_idx=routing_state.token_idx.detach(),
                        num_tokens=routing_state.num_tokens,
                        logits_std=logits_std,
                        logits_temp_std=logits_temp_std,
                        logits=logits.detach(),
                        expert_idx=routing_state.expert_idx.detach(),
                        weights=routing_state.weights.detach(),
                        rel_delta_sum=rel_delta_sum,
                        rel_delta_count=rel_delta_count,
                        expert_effective_rel_delta_sum=expert_effective_rel_delta_sum,
                        expert_effective_rel_delta_count=expert_effective_rel_delta_count,
                    )

                X = rearrange(
                    outputs.view(B, P, C_out, H_out, W_out), 
                    'b (h w) c ph pw -> b c (h ph) (w pw)',
                    h=h_patches, w=w_patches
                )
                X_sparse = X

                # Dense and residual block
                dens_in = merge_act(merge_gn(X))
                dense_delta = post_block(dens_in)
                X = X + dense_delta

                with torch.no_grad():
                    eps = 1e-8

                    sparse_norm = X_sparse.detach().flatten(1).norm(dim=1).mean()
                    dense_rel_delta = (
                        dense_delta.detach().flatten(1).norm(dim = 1).mean() / (sparse_norm + eps)
                    )

                    self.moe_aggregator.update_dense_layer(
                        layer_idx=layer_idx,
                        dense_rel_delta=float(dense_rel_delta.item()),
                        device=X.device,
                    )

                tot_z_loss += z_loss
                tot_balance_loss += balance_loss
                tot_overlap_loss += overlap_loss

        # Apply global pooling and prediction head
        x = self.pooler(X) 
        x = self.flatten(x) 
        logits = self.prediction_head(x) 
        
        tot_z_loss = tot_z_loss / E
        tot_balance_loss = tot_balance_loss / E
        tot_overlap_loss = tot_overlap_loss / E

        if collect_routes:
            return logits, tot_balance_loss, tot_overlap_loss, tot_z_loss, routing_debug

        return logits, tot_balance_loss, tot_overlap_loss, tot_z_loss,


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
