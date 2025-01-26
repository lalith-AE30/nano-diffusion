import torch
import numpy as np

from PIL import Image
from einops import rearrange
from network.unet import Unet
from sample import sample
from diffusion_schedules.scheduler import Schedules

def load_model_from_checkpoint(ckpt, device):
    ckpt = torch.load(ckpt, map_location=device, weights_only=True)
    model_config = ckpt["model_config"]
    betas = ckpt["betas"].cpu()

    model = Unet(**model_config["unet_params"]).to(device)
    model.load_state_dict(ckpt["model"])

    return (
        model,
        betas,
        model_config["image_size"],
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

model, betas, image_shape = load_model_from_checkpoint(
    "./models/blocks_cosine.pth", device
)
config = Schedules.create_from_betas(betas)
# sample 64 images
samples = sample(
    model,
    image_size=image_shape[1],
    batch_size=64,
    channels=image_shape[0],
    config=config,
)

ims = (samples[-1] + 1) * 0.5 * 255
ims = ims.clip(0, 255).astype(np.uint8)

Image.fromarray(
    (rearrange(ims, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=8, b2=8)).squeeze()
).save("./samples/Minecraft_samples.png")
