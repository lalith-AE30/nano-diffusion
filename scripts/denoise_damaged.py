import torch
import numpy as np
import torchvision.transforms.v2 as transforms
from PIL import Image
from einops import rearrange
from diffusion_schedules.scheduler import  Schedules
from sample import p_denoise_loop
from model import load_model_from_checkpoint

import logging
logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, betas, image_shape = load_model_from_checkpoint(
    "./models/blocks_cosine.pth", device
)
config = Schedules.create_from_betas(betas)

b1, b2 = 8, 8

img = Image.open('./noisy_image.png').convert("RGB")
img = transforms.functional.to_image(img).float().to(device)
img = ((img*1/255)*2-1).unsqueeze(0)
img = img.repeat([b1*b2, 1, 1, 1])

samples = p_denoise_loop(model, img, config, denoise_strength=0.3)

ims = (samples[-1] + 1) * 0.5 * 255
ims = ims.clip(0, 255).astype(np.uint8)

Image.fromarray(
    (rearrange(ims, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=b1, b2=b2)).squeeze()
).resize((256, 256), Image.NEAREST).save("./samples/Minecraft_samples.png")
