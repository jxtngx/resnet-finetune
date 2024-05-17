import os

from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Union

# ## get root path ## #
this_file = Path(__file__)
this_studio_idx = [
    i for i, j in enumerate(this_file.parents) if j.name.endswith("this_studio")
][0]
this_studio = this_file.parents[this_studio_idx]


@dataclass
class Config:
    cache_dir = os.path.join(this_studio, "data")
    log_dir = os.path.join(this_studio, "logs")
    ckpt_dir = os.path.join(this_studio, "checkpoints")
    prof_dir = os.path.join(this_studio, "logs", "profiler")
    perf_dir = os.path.join(this_studio, "logs", "perf")
    seed: int = 42


@dataclass
class ModuleConfig:
    model_name: str = "microsoft/resnet-50" # https://huggingface.co/microsoft/resnet-50
    learning_rate: float
    pixels_key: str = "pixel_values"
    label_key: str = "fine_label"
    loss_key: str = "loss"
    logits_key: str = "logits"
    hs_key: str = "hidden_states"
    learning_rate: float = 5e-05


@dataclass
class DataModuleConfig:
    dataset_name: str = "cifar100" # https://huggingface.co/datasets/cifar100
    num_classes: int = 100
    num_workers: int = cpu_count()
    batch_size: int = 16
    train_size: float = 0.8
    stratify_by_column: str = "fine_label"
    image_key: str = "img"
    train_split: str = "train"
    test_split: Optional[str] = None


@dataclass
class TrainerConfig:
    accelerator: str = "auto"  # Trainer flag
    devices: Union[int, str] = "auto"  # Trainer flag
    strategy: str = "auto"  # Trainer flag
    precision: Optional[str] = "16-mixed"  # Trainer flag
    max_epochs: int = 5  # Trainer flag
    deterministic: bool = False  # Trainer flag