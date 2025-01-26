import logging
import torch

from tqdm.auto import tqdm

from diffusion_schedules.scheduler import Schedules
from utils import extract

logger = logging.getLogger(__name__)


@torch.no_grad()
def p_sample(model, x, t, t_index, config: Schedules) -> torch.Tensor:
    betas_t = extract(config.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        config.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(config.sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(config.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, shape, config: Schedules):
    device = next(model.parameters()).device
    img = torch.randn(shape, device=device)
    return p_denoise_loop(model, img, config, 1.0)


@torch.no_grad()
def p_denoise_loop(model, img, config: Schedules, denoise_strength=1.0):
    device = next(model.parameters()).device

    b = img.shape[0]
    imgs = [img.cpu().numpy()]

    denoise_steps = int(denoise_strength * config.timesteps)

    for i in tqdm(
        reversed(range(0, denoise_steps)),
        desc="Sampling Progress",
        total=denoise_steps,
        disable=logger.level > logging.INFO,
    ):
        logger.debug(f"denoising step {i}")
        img = p_sample(
            model, img, torch.full((b,), i, device=device, dtype=torch.long), i, config
        ).clip(-1, 1)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(
    model,
    image_size,
    config: Schedules,
    batch_size=16,
    channels=3,
):
    return p_sample_loop(
        model, shape=(batch_size, channels, image_size, image_size), config=config
    )
