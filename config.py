from dataclasses import dataclass
import torch
import torch.nn.functional as F

from schedulers import cosine_beta_schedule

timesteps = 200

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

@dataclass
class Config:
    timesteps: int

    betas: torch.Tensor
    
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    
    posterior_variance: torch.Tensor