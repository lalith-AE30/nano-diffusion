from dataclasses import dataclass
import torch
import torch.nn.functional as F

from diffusion_schedules.schedule_curves import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
    SchedulerCurve
)

@dataclass
class Schedules:
    timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    @classmethod
    def create_from_betas(cls, betas: torch.Tensor):
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

        return cls(
            timesteps=betas.shape[0],
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            sqrt_recip_alphas=sqrt_recip_alphas,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            posterior_variance=posterior_variance,
        )

    @classmethod
    def create_schedule_from_scheduler(
        cls, timesteps: int = 200, scheduler: SchedulerCurve = SchedulerCurve.LINEAR
    ):
        # define beta schedule
        match scheduler:
            case SchedulerCurve.LINEAR:
                betas = linear_beta_schedule(timesteps=timesteps)
            case SchedulerCurve.COSINE:
                betas = cosine_beta_schedule(timesteps=timesteps)
            case SchedulerCurve.QUADRATIC:
                betas = quadratic_beta_schedule(timesteps=timesteps)
            case SchedulerCurve.SIGMOID:
                betas = sigmoid_beta_schedule(timesteps=timesteps)
            case _:
                raise NotImplementedError()

        return cls.create_from_betas(betas)
