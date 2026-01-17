import torch as t 
import pytest

from model.config import Config
from transformers import GPT2Model
from model.blocks import LayerNorm
from model.embeddings import Embedding, PositionEmbedding

# configs for all tests
cfg = Config()
batch_size = 16
sequence = 8

def test_embedding_shape_match():
    input = t.randint(low=0, high=cfg.d_vocab, size=(batch_size, sequence))
    expected_shape = [batch_size, sequence, cfg.d_model]

    embedding = Embedding(cfg)
    output = embedding(input)

    assert list(output.shape) == expected_shape

def test_embedding_parity():
    hf_model = GPT2Model.from_pretrained("gpt2")

    my_embed = Embedding(cfg)

    # grab weights from HF GPT2 
    pass