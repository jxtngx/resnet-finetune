from pathlib import Path
from typing import Optional, Union, Tuple

import torch
from transformers import ResNetForImageClassification
from transformers.modeling_outputs import ImageClassifierOutput

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import functional as F

from datamodule import image_processor
from config import Config, DataModuleConfig, ModuleConfig


class ImageClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = ModuleConfig.model_name,
        num_classes: int = DataModuleConfig.num_classes, 
        pixels_key: str = ModuleConfig.pixels_key,
        label_key: str = ModuleConfig.label_key,
        loss_key: str = ModuleConfig.loss_key,
        logits_key: str = ModuleConfig.logits_key,
        hs_key: str = ModuleConfig.hs_key,
        learning_rate: float = ModuleConfig.learning_rate,
    ) -> None:
        """a custom LightningModule for image classification

        Args:
            model_name: the name of the model and accompanying tokenizer
            num_classes: number of classes
            pixels_key: key used to access pixel values
            label_key: key used to access labels of model output
            loss_key: key used to access model return output
            logits_key: key used to access prediction tensor
            hs_key: key used to access hidden states of model output
            learning_rate: learning rate passed to optimizer
        """
        super().__init__()

        self.model_name = model_name
        self.model = ResNetForImageClassification.from_pretrained(model_name)
        self.accuracy = F.accuracy
        self.num_classes = num_classes
        self.pixels_key = pixels_key
        self.label_key = label_key
        self.loss_key = loss_key
        self.logits_key = logits_key
        self.hs_key = hs_key
        self.learning_rate = learning_rate


    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.model(
            batch[self.pixels_key],
            labels=batch[self.label_key],
        )
        self.log("train-loss", outputs[self.loss_key])
        return outputs[self.loss_key]

    def validation_step(self, batch, batch_idx) -> None:
        outputs = self.model(
            batch[self.pixels_key],
            labels=batch[self.label_key],
        )
        self.log("val-loss", outputs[self.loss_key], prog_bar=True)

        logits = outputs[self.logits_key]
        predicted_labels = torch.argmax(logits, 1)
        acc = self.accuracy(
            predicted_labels,
            batch[self.label_key],
            num_classes=self.num_classes,
            task="multiclass",
        )
        self.log("val-acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        outputs = self.model(
            batch[self.pixels_key],
            labels=batch[self.label_key],
        )

        logits = outputs[self.logits_key]
        predicted_labels = torch.argmax(logits, 1)
        acc = self.accuracy(
            predicted_labels,
            batch[self.label_key],
            num_classes=self.num_classes,
            task="multiclass",
        )
        self.log("test-acc", acc, prog_bar=True)

    def predict_step(
        self, 
        images, 
        cache_dir: Union[str, Path] = Config.cache_dir,
    ) -> str:
        batch = image_processor(images, model_name=self.model_name, cache_dir=cache_dir, image_key=None)
        # image_processor may cause tensors to lose device type and cause failure
        batch = batch.to(self.device)
        outputs = self.model(batch[self.pixels_key])
        logits = outputs[self.logits_key]
        return logits

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
