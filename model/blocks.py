import torch as t
import torch.nn as nn
import einops
import model.config as cfg
import math

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
    IGNORE: Float[Tensor, ""]

    def __init__(self, config: cfg.Config):
        super().__init__()
        self.cfg = config

        # weights and bias params
        self.W_Q = nn.Parameter(t.empty(config.n_heads, config.d_model, config.d_head))
        self.W_K = nn.Parameter(t.empty(config.n_heads, config.d_model, config.d_head))
        self.W_V = nn.Parameter(t.empty(config.n_heads, config.d_model, config.d_head))
        self.W_O = nn.Parameter(t.empty(config.n_heads, config.d_head, config.d_model))

        self.b_Q = nn.Parameter(t.zeros(config.n_heads, config.d_head))
        self.b_K = nn.Parameter(t.zeros(config.n_heads, config.d_head))
        self.b_V = nn.Parameter(t.zeros(config.n_heads, config.d_head))
        self.b_O = nn.Parameter(t.zeros(config.d_model))

        nn.init_normal_(self.W_Q, std=config.init_range)
        nn.init_normal_(self.W_K, std=config.init_range)
        nn.init_normal_(self.W_V, std=config.init_range)
        nn.init_normal_(self.W_O, std=config.init_range)

        # used in masking attention
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=cfg.DEVICE))

    def forward(self,
                normalized_resid_pre: Float[Tensor, "batch posn d_model"]
                ) -> Float[Tensor, "batch posn d_model"]:

        Q = einops.einsum(self.W_Q, normalized_resid_pre,
                          "n d h, b p d -> b p n h") + self.b_Q
        K = einops.einsum(self.W_K, normalized_resid_pre,
                          "n d h, b p d -> b p n h") + self.b_K
        V = einops.einsum(self.W_V, normalized_resid_pre,
                          "n d h, b p d -> b p n h") + self.b_V

        attn_scores = einops.einsum(Q, K, "b pq n h, b pk n h -> b n pq pk")/math.sqrt(self.config.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        z = einops.einsum(attn_scores,
                          V,
                          "b p n pq pk, b p n h -> b pq n h"
                          )
        
        result = einops.einsum(z, self.W_O, "b p n h, n h d -> b p n d")
        attn_out = einops.einsum(result, "b p n e -> b p e") + self.b_O

        return attn_out

    def apply_causal_mask(self, 
                          scores: Float[Tensor, "batch n_heads pos_q pos_k"]
                          ) -> Float[Tensor, "batch n_heads pos_q pos_k"]:
        """
        Applies a causal mask to scores and returns masked scores 
        """
        
        s = scores.shape[-1]
        ones = t.ones(s, s, device=scores.device)
        mask = t.triu(ones, diagonal=1).bool()
        
        return scores.masked_fill_(mask, self.IGNORE)




class MLP(nn.Module):
    def __init__(self, config: cfg.Config):
        super().__init__()
        self.config = config

        # params
        self.W_in = nn.Parameter(t.empty((config.d_model, config.d_mlp)))
        self.W_out = nn.Parameter(t.empty((config.d_mlp, config.d_model)))
        self.b_in = nn.Parameter(t.zeros((config.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((config.d_model)))

        nn.init.normal_(self.W_in, std=config.init_range)
        nn.init.normal_(self.W_out, std=config.init_range)


    def forward(self,
                normalized_resid_mid: Float[Tensor, "batch posn d_model"]
                ) -> Float[Tensor, "batch posn d_model"]:
        hidden = einops.einsum(normalized_resid_mid,
                               self.W_in,
                               "b p d, d m -> b p m") + self.b_in
        hidden = nn.functional.gelu(hidden)
        mlp_out = einops.einsum(hidden,
                                self.W_out,
                                "b p m, m d -> b p d") + self.b_out

        return mlp_out



