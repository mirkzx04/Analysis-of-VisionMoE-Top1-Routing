import torch

def clip_gradients(
    backbone_params,
    router_params,
    head_params = None,
    backbone_max_norm = 1.5,
    router_max_norm = 0.5,
    head_max_norm = 1.0,
):
    """
    Per-group gradient clipping. Call from on_after_backward, once the
    gradients have been computed.

    Clipping is done per group (not globally) so backbone and router can use
    different thresholds
    """
    torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=backbone_max_norm)
    torch.nn.utils.clip_grad_norm_(router_params, max_norm=router_max_norm)
    if head_params is not None:
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=head_max_norm)


def collect_model_prameters(model, collect_head = False):
    backbone_params, backbone_norm = [], []
    router_params = []
    router_pos_params = []
    head_params = []

    for n, p in model.named_parameters():
        is_router   = ('router_gate' in n) or ('router' in n) or ('gate' in n)
        is_position = ('position_w' in n) or ('position_scale' in n)
        is_head     = 'prediction_head' in n

        if collect_head and is_head:
            head_params.append(p)
            continue

        if is_router and is_position:
            router_pos_params.append(p)
        elif n.endswith('.bias') or 'norm' in n.lower() or 'bn' in n.lower():
            if is_router:
                router_params.append(p)
            else:
                backbone_norm.append(p)
        else:
            if is_router:
                router_params.append(p)
            else:
                backbone_params.append(p)

    backbone = backbone_params + backbone_norm
    router = router_params + router_pos_params

    if collect_head:
        return backbone, router, head_params
    return backbone, router

def collect_router_metrics(self, prefix, model) : 
    with torch.no_grad():
        router_metrics = model.moe_aggregator.finalize(include_layer_detail_metrics=False)

    log_dict = {}
    for key, value in router_metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        log_dict[f'{prefix}/{key}'] = float(value)
    return log_dict

