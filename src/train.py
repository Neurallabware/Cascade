"""Training entry point for CascadeTorch with Hydra configuration.

This script supports the standard lightning-hydra-template workflow:
    python src/train.py experiment=global_exc_30hz

CASCADE uses a unique training paradigm:
- For each noise level, an ensemble of models is trained
- Each model in the ensemble is independently initialized and trained
- The final prediction is the average of all ensemble models

This entry point handles both:
1. Standard Lightning training (single model, single noise level)
2. CASCADE-style multi-noise-level ensemble training
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.utils import (
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train the model.

    Args:
        cfg: DictConfig with all the configuration.

    Returns:
        Tuple of dicts with metrics and object outputs.
    """
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Instantiate datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiate callbacks
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Instantiate loggers
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiate trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log_hyperparameters(object_dict)

    # Train
    if cfg.get("train"):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    # Test
    if cfg.get("test"):
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training."""
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
