import torch
from network.unet import Unet

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