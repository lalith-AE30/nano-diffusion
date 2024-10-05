from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from sample import sample
from unet import Unet

import numpy as np
from einops import rearrange

from utils import extract, num_to_groups


# load dataset
class SpritesDataset(Dataset):
    def __init__(self):
        self.ims = np.load("sprites_1788_16x16.npy").astype(np.float32) / 255.0
        self.ims = rearrange(self.ims, "b h w c -> b c h w")
        self.ims = 2 * self.ims - 1

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        return self.ims[idx]


# dataset = SpritesDataset()
# image_size = 16
# batch_size = 32
# channels = 3

# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

from datasets import load_dataset
from torchvision.transforms import Compose
from torchvision import transforms

# load dataset from the hub
dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128


# define image transformations (e.g. using torchvision)
transform = Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)


# define function
def transforms(examples):
    examples["pixel_values"] = [
        transform(image.convert("L")) for image in examples["image"]
    ]
    del examples["image"]

    return examples


transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# create dataloader
dataloader = DataLoader(
    transformed_dataset["train"], batch_size=batch_size, shuffle=True
)

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

from config import timesteps, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod


# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


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


epochs = 5
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch['pixel_values'].shape[0]
        batch = batch['pixel_values'].to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, batch, t, loss_type="huber")

        if step % 100 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()

        # save generated images
        if step % save_and_sample_every == 0:
            milestone = (step + epoch * len(dataloader)) // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_list = list(
                map(
                    lambda n: torch.Tensor(
                        sample(model, image_size, batch_size=n, channels=channels)[-1]
                    ),
                    batches,
                )
            )
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(
                all_images, str(results_folder / f"sample-{milestone}.png"), nrow=6
            )

torch.save(model.state_dict(), "sprites_16x3_cosine_124.pth")
