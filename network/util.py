from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def Upsample(dim: int):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim: int):
    return nn.Conv2d(dim, dim, 4, 2, 1)
