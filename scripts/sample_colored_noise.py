import torch
import numpy as np
from PIL import Image
from einops import rearrange
from diffusion_schedules.scheduler import Schedules
from sample import p_denoise_loop
from model import load_model_from_checkpoint

import logging

logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, betas, image_shape = load_model_from_checkpoint(
    "../models/blocks_cosine.pth", device
)
config = Schedules.create_from_betas(betas)

b1, b2 = 8, 8

img = torch.randn(b1 * b2, 3, 16, 16).to(device)
img = (img + 1) * 0.5 * 255

# Color the noise
img[:, [0, 1]] *= 0.4


img = (img * 1 / 255) * 2 - 1
samples = p_denoise_loop(model, img, config)

ims = (samples[-1] + 1) * 0.5 * 255
ims = ims.clip(0, 255).astype(np.uint8)

Image.fromarray(
    (rearrange(ims, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=b1, b2=b2)).squeeze()
).resize((256, 256), Image.NEAREST).save("./samples/Minecraft_samples.png")
