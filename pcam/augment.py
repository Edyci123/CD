"""Batched GPU-side preprocessing and augmentation. Inputs are float [B,3,H,W] in [0,1]."""
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_float(xb_uint8_nhwc: torch.Tensor, device) -> torch.Tensor:
    """uint8 NHWC (CPU) -> float NCHW in [0,1] on device."""
    return xb_uint8_nhwc.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def geometric(x: torch.Tensor) -> torch.Tensor:
    """Random element of the dihedral group D4 per sample (rot90 x {0,1,2,3} and h-flip).
    Exact symmetries of the patch: the 32x32 label centre is preserved and no
    interpolation or padding is introduced."""
    b = x.shape[0]
    k = torch.randint(0, 4, (b,), device=x.device)
    out = x.clone()
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            out[m] = torch.rot90(x[m], r, dims=(2, 3))
    flip = torch.rand(b, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


AUGMENTATIONS = {
    "none": lambda x: x,
    "geo": geometric,
    # M3 / RQ1 adds "geo+color" (HED/HSV jitter) here.
}
