from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F

from utils import extract, to_rgb
from config import DiffusionConfig, create_diffusion_config


# forward diffusion
def q_sample(x_start, t, config: DiffusionConfig, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(config.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        config.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


from PIL import Image


def p_losses(
    denoise_model,
    x_start,
    t,
    config: DiffusionConfig,
    noise=None,
    loss_type="l1",
    debug=False,
):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, config=config, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if debug:
        print(x_noisy.max(), x_noisy.min())
        im = [
            rearrange(
                to_rgb(x_start),
                "b c h w-> h (b w) c",
            )
        ]
        im.append(
            rearrange(
                to_rgb(x_noisy),
                "b c h w-> h (b w) c",
            )
        )
        im.append(
            rearrange(
                to_rgb(predicted_noise),
                "b c h w-> h (b w) c",
            )
        )
        im.append(
            rearrange(
                to_rgb(noise),
                "b c h w-> h (b w) c",
            )
        )
        Image.fromarray(np.concatenate(im)).save("dbg.png")

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
