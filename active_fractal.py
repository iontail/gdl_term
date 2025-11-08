import torch
import torch.nn.functional as F

def mix_fractal(x, fractal_img, alpha: float = 0.2, active_lam: bool = False):
    B, C, H, W = x.shape
    N = len(fractal_img)
    device, dtype = x.device, x.dtype

    idx = torch.randint(0, N, (B,), device=device)

    overlay_imgs = []
    for i in idx:
        img_tensor, _ = fractal_img[int(i.item())]  # (tensor, label)
        img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=dtype)
        img_tensor = F.interpolate(img_tensor, size=(H, W), mode='bilinear', align_corners=False)
        overlay_imgs.append(img_tensor.squeeze(0))

    overlay = torch.stack(overlay_imgs, dim=0)  # (B, C, H, W)

    if active_lam:
        lam = torch.empty((B, 1), device=device, dtype=dtype).uniform_(0.1, 0.3)
    else:
        lam = torch.full((B, 1), alpha, device=device, dtype=dtype)

    mix_data = lam.view(B, 1, 1, 1) * overlay + (1 - lam.view(B, 1, 1, 1)) * x
    return mix_data, lam  # lam: (B, 1)
