# pip install -q -U einops datasets matplotlib tqdm

import os
import torch
import torch.nn.functional as F
from torch import nn  # , einsum
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    ToTensor,
    Lambda,
    ToPILImage,
    CenterCrop,
    Resize,
)

from pathlib import Path
from functools import partial

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image
from einops import rearrange

from sample import sample
from utils import exists, default, extract, num_to_groups
from resnet import ResnetBlock
from convnext import ConvNextBlock
from attention import Attention, LinearAttention
from schedulers import cosine_beta_schedule
from util_blocks import Residual, PreNorm, Downsample, Upsample


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


timesteps = 1000

# define beta schedule
betas = cosine_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = (
    betas
    * (1.0 - F.pad(alphas_cumprod[:-1], (1, 0), value=1.0))  # \bar{a}_t-1
    / (1.0 - alphas_cumprod)
)

# define image transformations (e.g. using torchvision)
transform = Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)

reverse_transform = Compose(
    [
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.0),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ]
)


# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t):
    # add noise
    x_noisy = q_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# load dataset
class SpritesDataset(Dataset):
    def __init__(self):
        self.ims = np.load("sprites_1788_16x16.npy").astype(np.float32) / 255.0
        self.ims = rearrange(self.ims, "b h w c -> b c h w")

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        return self.ims[idx]


dataset = SpritesDataset()
image_size = 16
batch_size = 128
channels = 3


results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4),
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

model_name = f"sprites_{image_size}x3_"

if os.path.exists("sprites_16x3_default_1248.pth"):
    model.load_state_dict(torch.load("sprites_16x3_default_1248.pth"))

# create dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

resume_training = True
if resume_training:
    epochs = 10
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="huber")

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

            # save generated images
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(
                    map(
                        lambda n: sample(model, batch_size=n, channels=channels),
                        batches,
                    )
                )
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(
                    all_images, str(results_folder / f"sample-{milestone}.png"), nrow=6
                )

    torch.save(model.state_dict(), "sprites_16x3_default_124.pth")

################################################
# Sampling from the model

# sample 64 images
samples = sample(model, image_size=image_size, batch_size=64, channels=channels)

Image.fromarray(
    (
        rearrange(samples[-1], "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=8, b2=8) * 255
    ).astype(np.uint8)
).save("diffusion.png")
