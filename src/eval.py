"""Evaluation/inference entry point for CascadeTorch.

This script supports:
1. Standard Lightning test evaluation
2. CASCADE-style inference: load pretrained models and predict spike rates

Usage:
    python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
    python src/eval.py experiment=global_exc_30hz
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

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.utils import (
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate the model.

    Args:
        cfg: DictConfig with configuration.

    Returns:
        Tuple of dicts with metrics and objects.
    """
    assert cfg.ckpt_path, "ckpt_path is required for evaluation"

    # Instantiate datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiate loggers
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiate trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    # Test
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation."""
    evaluate(cfg)


if __name__ == "__main__":
    main()
