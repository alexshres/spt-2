import datasets 

from dataclasses import dataclass
from transformer_lens.utils import tokenize_and_concatenate

MAIN = __name__ == "__main__"

def get_data(hf_data_name:str = "roneneldan/TinyStories", split:str = "train"):
    """Grabs TinyStories data"""
    return datasets.load_dataset(hf_data_name, split)


@dataclass
class TrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_for_epoch: int = 500
    lr: float = 0.001
    weight_decay: float = 1e-2
    wandb_project: str = "spt2"
    wandb_name: str | None = None



