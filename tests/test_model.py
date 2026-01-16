import torch as t 
import pytest

from transformers import GPT2Model
from model.blocks import LayerNorm
from model.embeddings import Embedding, PositionEmbedding



def test_embedding_shape_match():
    # TODO - Need to implement embedding test for making sure shape matches
    pass

def test_embedding_parity():
    # TODO - test if my embedding implementation matche GPT2 from HuggingFace
    pass