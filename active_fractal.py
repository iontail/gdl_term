import torch
import torch.nn.functional as F

def mix_fractal(x, fractal_batch, alpha: float = 0.2, active_lam: bool = False, retain_lam: bool = False):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    overlay = fractal_batch

    if active_lam:
        lam = torch.empty((B, 1), device=device, dtype=dtype).uniform_(0.15, 0.25)
    elif not retain_lam:
        lam = torch.full((B, 1), alpha, device=device, dtype=dtype)
    else:
        lam = torch.full((B, 1), 1.0, device=device, dtype=dtype)

    mix_data = lam.view(B, 1, 1, 1) * overlay + (1 - lam.view(B, 1, 1, 1)) * x
    return mix_data, lam  # lam: (B, 1)
