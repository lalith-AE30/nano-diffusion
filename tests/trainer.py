import torch
import logging
import torchvision.transforms.v2 as transforms

from pathlib import Path
from trainer import Trainer
from network.unet import Unet
from torch.optim import AdamW
from diffusion_schedules.schedule_curves import SchedulerCurve
from diffusion_schedules.scheduler import Schedules
from torchvision.datasets import CIFAR10


transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)


device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 32
channels = 3
batch_size = 64

model = Unet(
    dim=32,
    channels=channels,
    dim_mults=(1, 2, 4),
)
model.to(device)
model.compile()

optimizer = AdamW(model.parameters(), lr=1e-3)

timesteps = 1000
config = Schedules.create_schedule_from_scheduler(timesteps, SchedulerCurve.COSINE)


results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)
for file in results_folder.glob("*"):
    file.unlink()
save_and_sample_every = 300

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    betas=config.betas,
    dataset=dataset,
    batch_size=batch_size,
    results_folder=results_folder,
    save_and_sample_every=save_and_sample_every,
    device=device,
)

logging.basicConfig(level=logging.INFO)

trainer.train(5)