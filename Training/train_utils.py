import torch


def compute_patch_class(mask, grid_h, grid_w, num_classes, ignore_index=255):
    """Dominant (majority) non-ignore class per grid cell, for MI(expert, class)
    on segmentation.

    This MUST match the per-patch class definition used by
    ``Testing/experiments/mi_pos_class.py`` (the ``accumulate`` function) so the
    measured MI(exp, class) is comparable to the structural MI(pos, class)
    ceiling: a ``grid_h x grid_w`` grid over the (already eval-preprocessed) mask,
    the majority of the in-range ``[0, num_classes)`` pixels per cell ignoring
    ``ignore_index`` (255 void), and cells with no valid pixel are marked
    ``ignore_index`` (so they are excluded from the joint downstream). Ties are
    broken towards the lowest class index, matching ``np.bincount(...).argmax()``.

    Args:
        mask: ``[B, H, W]`` long label map at the model input resolution.
        grid_h, grid_w: patch grid (7x7 for the PCE model -> P = 49).
        num_classes: number of real classes (void/ignore is NOT one of them).
        ignore_index: void label to exclude from the majority vote (default 255).

    Returns:
        ``[B, grid_h * grid_w]`` long tensor, row-major so position
        ``p = i * grid_w + j`` matches the model's position index
        (``token_idx % P``); empty/void cells hold ``ignore_index``.
    """
    B, H, W = mask.shape
    ch, cw = H // grid_h, W // grid_w
    # Crop the bottom/right remainder so the reshape into cells is exact, exactly
    # like the integer-strided blocks in mi_pos_class.accumulate.
    m = mask[:, : grid_h * ch, : grid_w * cw]
    # [B, gh, ch, gw, cw] -> [B, gh, gw, ch*cw]: pixels grouped per grid cell.
    m = m.reshape(B, grid_h, ch, grid_w, cw).permute(0, 1, 3, 2, 4).reshape(
        B, grid_h, grid_w, ch * cw
    )
    valid = (m >= 0) & (m < num_classes)               # drops ignore_index (255) & negatives
    m_clamped = torch.where(valid, m, torch.zeros_like(m))
    onehot = torch.nn.functional.one_hot(m_clamped, num_classes).to(torch.float32)
    onehot = onehot * valid.unsqueeze(-1).to(onehot.dtype)
    hist = onehot.sum(dim=3)                            # [B, gh, gw, C] per-cell class counts
    # Lowest-index tie-break (counts are integers, so a <1 bias never flips a
    # genuine majority but resolves exact ties to the smaller class id).
    tie = torch.arange(num_classes, device=mask.device, dtype=hist.dtype) * 1e-6
    patch_class = (hist - tie).argmax(dim=-1)
    empty = hist.sum(dim=-1) == 0
    patch_class = torch.where(
        empty, torch.full_like(patch_class, ignore_index), patch_class
    )
    return patch_class.reshape(B, grid_h * grid_w).long()


def clip_gradients(
    backbone_params,
    router_params,
    head_params = None,
    post_block_params = None,
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
    # post_block separato dal backbone: clippato a parte con la stessa soglia.
    if post_block_params is not None:
        torch.nn.utils.clip_grad_norm_(post_block_params, max_norm=backbone_max_norm)


def collect_model_prameters(model, collect_head = False, collect_post_block = False):
    backbone_params, backbone_norm = [], []
    router_params = []
    router_pos_params = []
    head_params = []
    post_block_params = []

    for n, p in model.named_parameters():
        is_router     = ('router_gate' in n) or ('router' in n) or ('gate' in n)
        is_position   = ('position_w' in n) or ('position_scale' in n)
        is_head       = 'prediction_head' in n
        is_post_block = 'post_block' in n

        # post_block è separabile in un gruppo dedicato (di default resta nel backbone).
        if collect_post_block and is_post_block:
            post_block_params.append(p)
            continue

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

    result = [backbone, router]
    if collect_head:
        result.append(head_params)
    if collect_post_block:
        result.append(post_block_params)
    return tuple(result)

def collect_router_metrics(prefix, model) :
    with torch.no_grad():
        router_metrics = model.moe_aggregator.finalize(include_layer_detail_metrics=False)

    log_dict = {}
    for key, value in router_metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        log_dict[f'{prefix}/{key}'] = float(value)
    return log_dict

