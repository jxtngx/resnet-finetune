import os
from datetime import datetime
from time import perf_counter
from typing import Optional, Union

import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.profilers import PyTorchProfiler

from config import Config, DataModuleConfig, ModuleConfig, TrainerConfig
from datamodule import AutoImageProcessorDataModule
from module import ImageClassificationModule
from utils import log_perf, create_dirs

# Constants
model_name = ModuleConfig.model_name
dataset_name = DataModuleConfig.dataset_name

# Paths
cache_dir = Config.cache_dir
log_dir = Config.log_dir
ckpt_dir = Config.ckpt_dir
prof_dir = Config.prof_dir
perf_dir = Config.perf_dir
# creates dirs to avoid failure if empty dir has been deleted
create_dirs([cache_dir, log_dir, ckpt_dir, prof_dir, perf_dir])

# set matmul precision
# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


def train(
    accelerator: str = TrainerConfig.accelerator,  # Trainer flag
    devices: Union[int, str] = TrainerConfig.devices,  # Trainer flag
    strategy: str = TrainerConfig.strategy,  # Trainer flag
    precision: Optional[str] = TrainerConfig.precision,  # Trainer flag
    max_epochs: int = TrainerConfig.max_epochs,  # Trainer flag
    deterministic: bool = TrainerConfig.deterministic,  # Trainer flag
    lr: float = ModuleConfig.learning_rate,  # learning rate for LightningModule
    batch_size: int = DataModuleConfig.batch_size,  # batch size for LightningDataModule DataLoaders
    perf: bool = False,  # set to True to log training time and other run information
    profile: bool = False,  # set to True to profile. only use profiler to identify bottlenecks
) -> None:
    """a custom Lightning Trainer utility

    Note:
        for all Trainer flags, see https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    """

    # ######################### #
    # ## LightningDataModule ## #
    # ######################### #
    lit_datamodule = AutoImageProcessorDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )

    # ##################### #
    # ## LightningModule ## #
    # ##################### #
    lit_model = ImageClassificationModule(learning_rate=lr)

    # ################################################### #
    # ## Lightning Trainer callbacks, loggers, plugins ## #
    # ################################################### #

    # set logger
    logger = CSVLogger(
        save_dir=log_dir,
        name="csv-logs",
    )

    # set callbacks
    if perf:  # do not use EarlyStopping if getting perf benchmark
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="model",
            ),
        ]
    else:
        callbacks = [
            EarlyStopping(monitor="val-acc", mode="min"),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="model",
            ),
        ]

    # set profiler
    if profile:
        profiler = PyTorchProfiler(dirpath=prof_dir)
    else:
        profiler = None

    # ################################## #
    # ## create Trainer and call .fit ## #
    # ################################## #
    lit_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        deterministic=deterministic,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        log_every_n_steps=50,
    )
    start = perf_counter()
    lit_trainer.fit(model=lit_model, datamodule=lit_datamodule)
    stop = perf_counter()

    # ## log perf results ## #
    if perf:
        log_perf(start, stop, perf_dir, lit_trainer)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(train, as_positional=False)
