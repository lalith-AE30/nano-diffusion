import torch
from torch import nn
<<<<<<< HEAD
import math
=======
import numpy as np
>>>>>>> d245aaf263e5ee941acf8607e576e89c31b949a0

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
<<<<<<< HEAD
        embeddings = math.log(10000.0) / (half_dim - 1)
=======
        embeddings = np.log(10000.0) / (half_dim - 1)
>>>>>>> d245aaf263e5ee941acf8607e576e89c31b949a0
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
