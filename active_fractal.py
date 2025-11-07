import torch
from torch import transforms

def mix_fractal(x, fractal_img, alpha: float = 0.2, active_lam: bool = False):
    # fractal_data: list[PIL.Image]
    B, C, H, W = x.shape
    N = len(fractal_img)
    device, dtype = x.device, x.dtype

    idx = torch.randint(0, N, (B,)) 

    to_tensor = transforms.ToTensor()
    overlay_imgs = [to_tensor(fractal_img[i]).resize_((C, H, W)) for i in idx]
    overlay = torch.stack(overlay_imgs).to(device=device, dtype=dtype)

    if active_lam:
        lam = torch.empty(B, 1, 1, 1, device=device, dtype=dtype).uniform_(0.1, 0.3) # setting the range
    else:
        lam = torch.full((B, 1, 1, 1), alpha, device=device, dtype=dtype)

    mix_data = lam * overlay + (1 - lam) * x
    return mix_data, lam

def mix_fractal_criterion(criterion, preds, targets, lam):
    return criterion(preds, targets) * (1-lam)