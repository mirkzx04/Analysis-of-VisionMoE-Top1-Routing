# === HYPER PARAMETERS SCHEDULERS ===
def temp_scheduler(self, current_epoch, router_start_epoch, router_warmup, train_epochs, temp_epochs, temp_init, temp_final):
    e  = int(current_epoch)
    t0 = int(router_start_epoch)
    tw = int(router_warmup)
    T  = int(train_epochs)

    t_start = t0 + tw
    t_end   = min(t_start + int(temp_epochs), T)

    if e < t_start:
        return float(temp_init)

    if e < t_end:
        progress   = (e - t_start) / float(max(1, t_end - t_start))
        progress   = min(max(progress, 0.0), 1.0)
        cosine_dec = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return float(temp_final) + (float(temp_init) - float(temp_final)) * cosine_dec

    return float(self.temp_final)

def noise_scheduler(self, current_epoch, router_start_epoch, router_warmup, train_epochs, temp_epochs):
    e = int(current_epoch)
    rs = int(router_start_epoch)
    rw = int(router_warmup)
    T = int(train_epochs)

    t_start = rs + rw
    t_end = min(t_start + int(temp_epochs), T)

    noise_max = 0.02
    noise_final = 0.0

    # Prima che il router sia attivo: niente rumore
    if e < rs:
        return 0.0

    # Plateau di esplorazione ad alto rumore
    if e < t_start:
        return noise_max

    if e < t_end :
        progress = (e - t_start) / float(max(1, t_end - t_start))
        progress = min(max(progress, 0.0), 1.0)
        cosine_dec = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return noise_final + (noise_max - noise_final) * cosine_dec

    return noise_final

# === LEARNING RAGE SCHEDULERS ===
def backbone_lr_lambda(epoch, warmup_backbone):
    e = int(epoch)
    if  e < warmup_backbone:
    # warmup 0 -> 1
        return (e + 1) / float(max(1, warmup_backbone))

    eta_min = 0.1

    # cosine decay da warmup_epochs a T
    progress = (e - warmup_backbone) / float(max(1, T - warmup_backbone))
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return eta_min + (1.0 - eta_min) * cos            # 1 -> eta_min0

def router_lr_lambda(epoch, router_start_epoch, warmup_router):
    e = int(epoch)
    # Router spento fino a router_start_epoch
    if e < router_start_epoch:
        return 0.0

    # Warmup router: 0 -> 1 in router_warmup epoche
    if e < router_start_epoch + warmup_router:
        pct = (e - router_start_epoch + 1) / float(max(1, warmup_router))
        return pct

    # Cosine decay da (router_start_epoch + router_warmup) a T
    eta_min_r = 0.10  # router spesso ok un po' più alto

    progress = (e - router_start_epoch) / float(max(1, T - router_start_epoch))
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return eta_min_r + (1.0 - eta_min_r) * cos
