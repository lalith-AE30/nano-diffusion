import torch
import numpy as np

from PIL import Image
from einops import rearrange
import torchvision.transforms.v2 as transforms
from diffusion_schedules.scheduler import Schedules
from sample import p_denoise_loop
from utils import load_model_from_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

model, betas, image_shape = load_model_from_checkpoint(
    "./models/blocks_cosine.pth", device
)
config = Schedules.create_from_betas(betas)

img = Image.open('./noisy_image.png').convert("RGB")
img = transforms.functional.to_image(img).float()
img = ((img*1/255)*2-1).unsqueeze(0)

samples = p_denoise_loop(model, img, config)

ims = (samples[-1] + 1) * 0.5 * 255
ims = ims.clip(0, 255).astype(np.uint8)

Image.fromarray(
    (rearrange(ims, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=1, b2=1)).squeeze()
).save("./samples/Minecraft_samples.png")
