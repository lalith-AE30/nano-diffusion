import torch
from pathlib import Path
import torchvision.transforms.v2 as transforms

from datasets import BlocksDataset
from schedule_curves import cosine_beta_schedule
from trainer import OptimizerType, Trainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.ToPureTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ]
)

# dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
dataset = BlocksDataset("./data/block", transform=transform)

results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)
for file in results_folder.glob("*"):
    file.unlink()
save_and_sample_every = 300

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "blocks_cosine.pth"

if Path(model_path).exists():
    trainer = Trainer.from_checkpoint(model_path, dataset, device=device)
    trainer.results_folder = results_folder
    for g in trainer.optimizer.param_groups:
        g['lr'] = 2e-4
else:
    trainer = Trainer(
        model_config={
            "unet_params": {
                "dim": 32,
                "dim_mults": (1, 2, 4, 8),
                "channels": 3,
            }
        },
        optimizer_config={
            "type": OptimizerType.ADAM.value,
            "params": {"lr": 4e-4},
        },
        betas=cosine_beta_schedule(1000),
        dataset=dataset,
        results_folder=results_folder,
        save_and_sample_every=save_and_sample_every,
        device=device,
    )

print(sum(torch.numel(p) for p in trainer.model.parameters()))

try:
    trainer.train(batch_size=256, epochs=5000)
except KeyboardInterrupt:
    logger.info("Training interrupted, saving model...")

trainer.save_checkpoint(model_path)
