import torch

from config import create_diffusion_config
from unet import Unet

import numpy as np
from einops import rearrange
from PIL import Image

from sample import sample

image_size = 28
channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4),
)
model.to(device)

# load model
model.load_state_dict(torch.load("fashion_28x1_linear_124.pth", map_location=device))

# sample 64 images
samples = sample(
    model,
    image_size=image_size,
    batch_size=64,
    channels=channels,
    config=create_diffusion_config(200),
)

ims = (samples[-1] + 1) * 0.5 * 255
ims = ims.clip(0, 255).astype(np.uint8)

Image.fromarray(
    (rearrange(ims, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=8, b2=8)).squeeze(), mode="L"
).save("diffusion.png")