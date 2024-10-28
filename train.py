from pathlib import Path
from einops import rearrange
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import InterpolationMode

from sample import sample
from network.unet import Unet
from forward import p_losses

from schedulers import Scheduler
from config import create_diffusion_config
from utils import num_to_groups

import numpy as np
from torchvision.datasets import CIFAR10


# load dataset
class SpritesDataset(Dataset):
    def __init__(self):
        self.ims = np.load("sprites_1788_16x16.npy").astype(np.float32)
        self.ims = (self.ims / 255) * 2 - 1
        self.ims = rearrange(self.ims, "b h w c -> b c h w")
        self.ims = torch.Tensor(self.ims)
        # self.ims = transforms.functional.resize(self.ims, (32, 32), InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        return self.ims[idx], 0

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])

# dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset = SpritesDataset()
image_size = 16
channels = 3
batch_size = 128

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)
for file in results_folder.glob("*"):
    file.unlink()
save_and_sample_every = 300

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=32,
    channels=channels,
    dim_mults=(1, 2, 4),
)
model.to(device)

# model = torch.compile(model)

models_path = Path('models')
model_name = "exp(dim=32)_sprites_16x3_linear_124_320.pth"

if (models_path / model_name).exists():
    model.load_state_dict(torch.load(models_path / model_name, map_location=device))
    print("Found checkpoint, resuming training...")

optimizer = AdamW(model.parameters(), lr=1e-3)

timesteps = 320
config = create_diffusion_config(timesteps, Scheduler.LINEAR)

epochs = 20
try:
    for epoch in range(epochs):
        for step, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(
                model, batch, t, config, loss_type="huber", debug=(step % 100 == 0)
            )

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
                            sample(
                                model,
                                image_size,
                                batch_size=n,
                                channels=channels,
                                config=config,
                            )[-1]
                        ),
                        batches,
                    )
                )
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(
                    all_images, str(results_folder / f"sample-{milestone}.png"), nrow=6
                )
except KeyboardInterrupt:
    print("KeyboardInterrupt: save model?")
    if input() in ["N", "n"]:
        exit(0)
except Exception as e:
    print(f"Error: {e}")
    exit(1)

torch.save(model.state_dict(), models_path / model_name)
