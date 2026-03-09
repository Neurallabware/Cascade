"""Utility functions for the lightning-hydra-template training pipeline.

Based on the standard lightning-hydra-template utilities, adapted for CascadeTorch.
"""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Apply optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        print_config_tree(cfg, resolve=True, save_to_file=True)


def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided."""
    if not cfg.get("tags"):
        if "id" in cfg:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Skipping tag enforcement.")


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when exception is raised.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder
        - etc. (adjust depending on your needs)

    Example:
        @task_wrapper
        def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            ...
            return metric_dict, object_dict
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")
            raise
        finally:
            # Display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir if cfg.get('paths') else 'N/A'}")

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule."""
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Any]:
    """Instantiate callbacks from config."""
    callbacks: List[Any] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            import hydra
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Any]:
    """Instantiate loggers from config."""
    loggers: List[Any] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            import hydra
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Log hyperparameters to all loggers. Controls which cfg parts are saved."""
    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def print_config_tree(
    cfg: DictConfig,
    print_order: List[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using Rich library."""
    try:
        from rich.tree import Tree
        from rich import print as rprint
        from omegaconf import OmegaConf

        tree = Tree("CONFIG")

        queue = []
        # add fields from print_order first
        for field in print_order:
            if field in cfg:
                queue.append(field)
        # then add remaining fields
        for field in cfg:
            if field not in queue:
                queue.append(field)

        for field in queue:
            branch = tree.add(field)
            config_group = cfg[field]
            if isinstance(config_group, DictConfig):
                branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
            else:
                branch_content = str(config_group)
            branch.add(branch_content)

        rprint(tree)

        if save_to_file:
            import os
            output_dir = cfg.get("paths", {}).get("output_dir", ".")
            if output_dir and output_dir != ".":
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "config_tree.log"), "w") as file:
                    from rich.console import Console
                    console = Console(file=file, width=200)
                    console.print(tree)
    except ImportError:
        log.warning("Rich not installed. Skipping config tree printing.")
