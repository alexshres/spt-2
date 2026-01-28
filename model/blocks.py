import torch as t
import torch.nn as nn
import einops
import model.config as cfg

from torch import Tensor
from jaxtyping import Float

class LayerNorm(nn.Module):
    """
    Performs Layer Normalization on a batch passsed in

    Steps:
        * make mean 0
        * normalize to have variance 1
        * scale with learned weights (\\gamma in PyTorch doc)
        * shift with learned bias    (\\beta in PyTorch doc)
    """

    def __init__(self, config: cfg.Config):
        super().__init__()
        self.cfg = config

        # scale mean with learned weights
        self.w = nn.Parameter(t.ones(config.d_model))
        # shift mean w/ learned bias
        self.b = nn.Parameter(t.zeros(config.d_model))

    def forward(self, 
                residual: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:  # noqa: F722

        # can also do
        # >>> residual.mean(dim=-1, keepdim=True)
        mean = einops.reduce(residual, "b s d -> b s 1", "mean")
        var = residual.var(dim=-1, keepdim=True, unbiased=False)

        scale = (var + self.cfg.ln_eps).sqrt()
        centered = (residual-mean)/scale

        ln = centered*self.w + self.b

        if self.cfg.debug:
            print(f"Layer Norm: {ln.shape}")

        return ln

class CausalAttention(nn.Module):
    pass


class MLP(nn.Module):
    pass



