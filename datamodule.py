import os
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union, Optional

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from PIL.PngImagePlugin import PngImageFile

import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from config import Config, DataModuleConfig, ModuleConfig


class AutoImageProcessorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = DataModuleConfig.dataset_name,
        cache_dir: Union[str, Path] = Config.cache_dir,
        model_name: str = ModuleConfig.model_name,
        num_classes: int = DataModuleConfig.num_classes,
        columns: list = ["pixel_values", "fine_label"],
        batch_size: int = DataModuleConfig.batch_size,
        train_size: float = DataModuleConfig.train_size,
        stratify_by_column: str = DataModuleConfig.stratify_by_column,
        image_key: str = DataModuleConfig.image_key,
        train_split: str = DataModuleConfig.train_split,
        test_split: Optional[str] = DataModuleConfig.test_split,
        num_workers: int = DataModuleConfig.num_workers,
        seed: int = Config.seed,
    ) -> None:
        """a custom LightningDataModule for Hugging Face datasets requiring image processing

        Args:
            dataset_name: the name of the dataset as given on HF datasets
            cache_dir: corresponds to HF datasets.load_dataset
            model_name: the name of the model and accompanying tokenizer
            num_classes: the number of labels
            columns: the list of column names to pass to the HF dataset's .set_format method
            batch_size: the batch size to pass to the PyTorch DataLoaders
            train_size: the size of the training data split to pass to .train_test_split
            stratify_by_column: column name of labels to be used to perform stratified split of data
            train_split: the name of the training split as given on HF Hub
            test_split: the name of the test split as given on HF Hub
            num_workers: corresponds to torch.utils.data.DataLoader
            seed: the seed used in lightning.pytorch.seed_everything
        """
        super().__init__()

        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.train_size = train_size
        self.train_split = train_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.num_classes = num_classes
        self.columns = columns
        self.stratify_by_column = stratify_by_column
        self.image_key = image_key

    def prepare_data(self) -> None:
        pl.seed_everything(self.seed)
        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        cache_dir_is_empty = len(os.listdir(self.cache_dir)) == 0

        if cache_dir_is_empty:
            rank_zero_info(f"[{str(datetime.now())}] Downloading dataset.")
            dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
            rank_zero_info(
                f"[{str(datetime.now())}] Processing dataset. This may take several moments"
            )
            dataset = dataset.map(
                image_processor,
                batched=True,
                fn_kwargs={
                    "model_name": self.model_name,
                    "image_key": self.image_key,
                    "cache_dir": self.cache_dir,
                },
            )
        else:
            rank_zero_info(
                f"[{str(datetime.now())}] Data cache exists. Loading from cache."
            )

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            # load and split
            dataset = load_dataset(
                self.dataset_name,
                split=self.train_split,
                cache_dir=self.cache_dir,
            )
            dataset = dataset.train_test_split(
                train_size=self.train_size,
                stratify_by_column=self.stratify_by_column,
                seed=self.seed,
            )
            # prep train
            self.train_data = dataset["train"].map(
                image_processor,
                batched=True,
                fn_kwargs={
                    "model_name": self.model_name,
                    "image_key": self.image_key,
                    "cache_dir": self.cache_dir,
                },
            )
            self.train_data.set_format("torch", columns=self.columns)
            # prep val
            self.val_data = dataset["test"].map(
                image_processor,
                batched=True,
                fn_kwargs={
                    "model_name": self.model_name,
                    "image_key": self.image_key,
                    "cache_dir": self.cache_dir,
                },
            )
            self.val_data.set_format("torch", columns=self.columns)
            # free mem from unneeded dataset obj
            del dataset
        if stage == "test" or stage is None:
            if not getattr(self, "val_data", None):
                # load and split
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.train_split,
                    cache_dir=self.cache_dir,
                )
                dataset = dataset.train_test_split(
                    train_size=self.train_size,
                    stratify_by_column=self.stratify_by_column,
                    seed=self.seed,
                )
                # prep val
                self.val_data = dataset["test"].map(
                    image_processor,
                    batched=True,
                    fn_kwargs={
                        "model_name": self.model_name,
                        "cache_dir": self.cache_dir,
                    },
                )
                self.val_data.set_format("torch", columns=self.columns)
                # free mem from unneeded dataset obj
                del dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )


def image_processor(
    batch: dict,
    *,
    model_name: str,
    image_key: str,
    cache_dir: Union[str, Path],
):

    
    image_processor = AutoImageProcessor.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    images = (
        batch if isinstance(batch, PngImageFile) else batch[image_key]
    )  # allow for inference input as raw text
    return image_processor(images, return_tensors="pt")
