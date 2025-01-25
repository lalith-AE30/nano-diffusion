from enum import Enum
import logging
from pathlib import Path

from einops import rearrange
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from scheduler import Schedules
from sample import sample
from network.unet import Unet
from forward import p_losses

from utils import num_to_groups, to_rgb

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class Trainer:
    def __init__(
        self,
        model_config: dict,
        optimizer_config: dict,
        betas: torch.Tensor,
        dataset,
        logging_interval: int = 100,
        results_folder=Path("./results"),
        save_and_sample_every: int = 300,
        device=torch.device("cpu"),
    ):
        to_compile = model_config.get("compile", False)
        data_shape = model_config.get("image_size", None)
        self.model = Unet(**model_config["unet_params"]).to(device)
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.betas = betas
        self.config = Schedules.create_from_betas(betas)
        self.dataset = dataset
        # self.batch_size = batch_size
        self.logging_interval = logging_interval
        self.results_folder = results_folder
        self.save_and_sample_every = save_and_sample_every
        self.device = device
        self.epochs = 0

        match optimizer_config["type"]:
            case OptimizerType.ADAM.value:
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), **optimizer_config["params"]
                )
            case OptimizerType.ADAMW.value:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), **optimizer_config["params"]
                )
            case OptimizerType.SGD.value:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), **optimizer_config["params"]
                )
        if data_shape and data_shape != self.dataset.data[0].shape:
            return ValueError("Data shape does not match model input shape")
        if not data_shape:
            self.model_config["image_size"] = self.dataset.data[0].shape
        if to_compile:
            self.model.compile()
        if self.dataset.data[0].shape[0] == self.dataset.data[0].shape[1]:
            self.image_size = self.dataset.data[0].shape[0]
        else:
            return ValueError("Image is not square")
        if len(self.dataset.data[0].shape) > 2:
            self.channels = self.dataset.data[0].shape[2]
        else:
            self.channels = 1

    def train(self, batch_size, epochs):
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        for _ in range(epochs):
            for step, (batch, _) in enumerate(dataloader):
                self.optimizer.zero_grad()

                batch = batch.to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(
                    0, self.config.timesteps, (batch_size,), device=self.device
                ).long()

                loss, *debug_data = p_losses(
                    self.model,
                    batch,
                    t,
                    self.config,
                    loss_type="huber",
                    debug=(step % self.logging_interval == 0),
                )
                loss.backward()
                self.optimizer.step()

                if step % self.logging_interval == 0:
                    logger.info(f"loss: {loss.item()}")
                    self.save_debug_image(*debug_data)

                # save generated images
                if (
                    self.save_and_sample_every != 0
                    and step % self.save_and_sample_every == 0
                ):
                    milestone = (
                        step + self.epochs * len(dataloader)
                    ) // self.save_and_sample_every
                    self.save_sample(batch_size, milestone)
            self.epochs += 1

    def save_checkpoint(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": self.model_config,
                "optimizer": self.optimizer.state_dict(),
                "optimizer_config": self.optimizer_config,
                "epochs": self.epochs,
                "betas": self.betas,
            },
            path,
        )

    def save_debug_image(
        self, x_start, x_noisy, predicted_noise, noise, save_path="./debug.png"
    ):
        im = [
            rearrange(
                to_rgb(x_start),
                "b c h w-> h (b w) c",
            )
        ]
        im.append(
            rearrange(
                to_rgb(x_noisy),
                "b c h w-> h (b w) c",
            )
        )
        im.append(
            rearrange(
                to_rgb(predicted_noise - noise),
                "b c h w-> h (b w) c",
            )
        )
        im.append(
            rearrange(
                to_rgb(noise - noise),
                "b c h w-> h (b w) c",
            )
        )
        Image.fromarray(np.concatenate(im)).save(save_path)

    def save_sample(self, batch_size, milestone):
        batches = num_to_groups(4, batch_size)
        all_images_list = list(
            map(
                lambda n: torch.Tensor(
                    np.array(
                        sample(
                            self.model,
                            self.image_size,
                            batch_size=n,
                            channels=self.channels,
                            config=self.config,
                        )
                    )
                ),
                batches,
            )
        )
        all_images = torch.cat(list(im[-1, :] for im in all_images_list), dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(
            all_images,
            str(self.results_folder / f"sample-{milestone}.png"),
            nrow=6,
        )
        idx = torch.linspace(0, self.config.timesteps - 1, 20).long()
        Image.fromarray(
            to_rgb(rearrange(all_images_list[0][idx, :], "t b c h w -> (b h) (t w) c"))
        ).save(str(self.results_folder / f"sample-process-{milestone}.png"))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        dataset,
        device,
    ):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model_config = ckpt["model_config"]
        optimizer_config = ckpt["optimizer_config"]
        betas = ckpt["betas"].cpu()

        trainer = cls(
            model_config=model_config,
            optimizer_config=optimizer_config,
            betas=betas,
            dataset=dataset,
            device=device,
        )

        trainer.model.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.epochs = ckpt["epochs"]

        return trainer
