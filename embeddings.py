import torch as t
import torch.nn as nn
import einops
import config as cfg

from torch import Tensor
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(self, config: cfg.Config):
        self.cfg = config
        
        # fill the learnable embedding matrix with a normal distribution
        # and standard deviation from our config class
        self.W_E = nn.Parameter(t.empty((config.d_vocab, config.d_model)))
        nn.init.normal_(self.W_E, std=config.init_range)

    def forward(self,
                tokens: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq d_model"]:
        raise NotImplementedError


class PositionEmbedding(nn.Module):
    def __init__(self, config: cfg.Config):
        self.cfg = config

        self.W_P = nn.Parameter(t.empty((config.n_ctx, config.d_model)))
        nn.init.normal_(self.W_P, std=config.init_range)

    def forward(self, tokens):
        raise NotImplementedError