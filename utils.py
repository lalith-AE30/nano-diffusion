from inspect import isfunction
import numpy as np
import torch

def exists(x: any) -> bool:
    return x is not None


def default(val: any, d: any) -> any:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def to_latent(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x).unsqueeze(0) / 255 * 2 - 1

def to_rgb(x: torch.Tensor) -> np.ndarray:
    return ((x + 1) * 0.5 * 255).detach().cpu().numpy().clip(0, 255).astype(np.uint8)