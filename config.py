import dataclasses


@dataclasses.dataclass
class Config:
    d_model: int = 768          # length of embedding vector
    debug: bool = True
    ln_eps: float = 1e-5        # prevents division by zero error
    d_vocab: int = 50257        # number of total unique tokens in GPT2
    init_range: float = 0.02    # cargo cult
    n_ctx: int = 1024
    d_head: int = 64            # project into d_head dimension in Q, K, V space
    d_mlp: int = 3027           # 4*d_model
    n_heads: int = 12           # heads in MHA
    n_layers: int = 12          # number of layers (MHA + MLP)
