from torch import nn

from config import DiffusionConfig
from forward import p_losses
from network.unet import Unet

class DDPM(nn.Module):
    def __init__(self, config: DiffusionConfig, *unet_args, **unet_kwargs):
        super().__init__()

        self.config = config
        self.unet = Unet(*unet_args, **unet_kwargs)

    def forward(self, x, timesteps):
        loss = p_losses(self.unet, x, timesteps, loss_type="huber")