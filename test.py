from pathlib import Path
import torch

from config import create_diffusion_config
from network.unet import Unet

import numpy as np
from einops import rearrange
from PIL import Image

from sample import sample
from schedulers import Scheduler

model_name = "sprites_16x3_cosine_122_200.pth"

_, ic, sch, mul, timesteps = model_name[:model_name.find('.')].split('_')

timesteps = int(timesteps)

image_size, channels = map(int, ic.split('x'))
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(int(i) for i in mul),
)
model.to(device)
if 'compiled' in model_name:
    model = torch.compile(model)

match sch:
    case "linear":
        config = create_diffusion_config(timesteps, Scheduler.LINEAR)
    case "cosine":
        config = create_diffusion_config(timesteps, Scheduler.LINEAR)
    case "quadratic":
        config = create_diffusion_config(timesteps, Scheduler.QUADRATIC)
    case "sigmoid":
        config = create_diffusion_config(timesteps, Scheduler.SIGMOID)
    case _:
        raise NotImplementedError()

# load model
model.load_state_dict(torch.load(Path('models') / model_name, map_location=device))

# sample 64 images
samples = sample(
    model,
    image_size=image_size,
    batch_size=64,
    channels=channels,
    config=config,
)

ims = (samples[-1] + 1) * 0.5 * 255
ims = ims.clip(0, 255).astype(np.uint8)

Image.fromarray((rearrange(ims, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=8, b2=8)).squeeze()).save("sprites_sample.png")
