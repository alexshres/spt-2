import torch as t
import torch.nn as nn
import einops
import model.config as cfg

from torch import Tensor
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(self, config: cfg.Config):
        super().__init__()
        self.cfg = config
        
        # fill the learnable embedding matrix with a normal distribution
        # and standard deviation from our config class
        self.W_E = nn.Parameter(t.empty((config.d_vocab, config.d_model)))
        nn.init.normal_(self.W_E, std=config.init_range)

    def forward(self,
                tokens: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq d_model"]:
        """
        Arguments:
            tokens (Int[Tensor, "batch seq"]):  Batches of sequences

        Returns:
            embed (Float[Tensor, "batch seq d_model"]): Batches of sequences projected onto embedding space (d_model)

        """

        return self.W_E[tokens, :]


class PositionEmbedding(nn.Module):
    def __init__(self, config: cfg.Config):
        super().__init__()
        self.cfg = config

        self.W_P = nn.Parameter(t.empty((config.n_ctx, config.d_model)))
        nn.init.normal_(self.W_P, std=config.init_range)

    def forward(self, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq d_model"]:
        """
        Args:
            tokens (Int[Tensor, "batch seq"]): Batches of sequences

        Returns:
            pos_embed (Float[Tensor, "batch seq d_model"]): Batches of sequences projected onto positional embedding space
        """

        seq_len = tokens.shape[1]
        pos_embed = self.W_P[:seq_len, :]
        pos_embed = einops.repeat(pos_embed,
                                  "pos d_model -> batch pos d_model")

        return pos_embed