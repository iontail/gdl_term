import torch
import numpy as np
import torch.nn.functional as F

from torchvision import transforms

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


#########################
# below are for experiments
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