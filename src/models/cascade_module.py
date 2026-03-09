"""PyTorch Lightning module wrapping CascadeNet for spike inference.

Training logic extracted from cascade2p/cascade.py -- scientific logic preserved exactly.
Prediction logic replicates cascade2p/cascade.py predict() function.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim import Optimizer

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L  # type: ignore[no-redef]

from torchmetrics import MeanSquaredError

from src.models.components.cascade_net import CascadeNet
from src.utils.model_zoo import get_model_paths
from src.utils.noise import calculate_noise_levels


class CascadeLitModule(L.LightningModule):
    """Lightning module for training and evaluating CascadeNet.

    Parameters
    ----------
    net : CascadeNet
        The neural network model.
    optimizer : functools.partial[Optimizer]
        A partial function that creates an optimizer when called with
        ``params`` as keyword argument. Example:
        ``functools.partial(torch.optim.Adagrad, lr=0.05)``.
    scheduler : functools.partial or None, optional
        A partial function that creates a learning rate scheduler when
        called with ``optimizer`` as keyword argument.
    """

    def __init__(
        self,
        net: CascadeNet,
        optimizer: partial[Optimizer],
        scheduler: partial | None = None,
    ) -> None:
        super().__init__()

        # Save hyperparams but skip the net object itself (it's a Module)
        self.save_hyperparameters(ignore=["net"])

        self.net = net
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler

        # Loss function -- MSE, matching the original training
        self.criterion = nn.MSELoss()

        # Metrics for each stage
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, windowsize, 1)``.

        Returns
        -------
        torch.Tensor
            Predicted spike rates of shape ``(batch, 1)``.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Log initial validation metrics as zero at epoch 0."""
        self.val_mse.reset()

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute predictions and loss for a batch.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            ``(X, Y)`` where X has shape ``(batch, windowsize, 1)`` and
            Y has shape ``(batch, 1)``.

        Returns
        -------
        tuple of (loss, predictions, targets)
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, y_hat, y = self._shared_step(batch)

        # Log metrics
        self.train_mse(y_hat, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, y_hat, y = self._shared_step(batch)

        # Log metrics
        self.val_mse(y_hat, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, y_hat, y = self._shared_step(batch)

        # Log metrics
        self.test_mse(y_hat, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and optional learning rate scheduler."""
        optimizer = self.optimizer_partial(params=self.parameters())

        config: dict[str, Any] = {"optimizer": optimizer}

        if self.scheduler_partial is not None:
            scheduler = self.scheduler_partial(optimizer=optimizer)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }

        return config


def predict(
    model_name: str,
    traces: NDArray[np.floating],
    model_folder: str = "Pretrained_models",
    threshold: int | bool = 0,
    padding: float = np.nan,
    trace_noise_levels: NDArray[np.floating] | None = None,
    verbosity: int = 1,
    device: torch.device | None = None,
) -> NDArray[np.floating]:
    """Predict spiking activity from calcium traces using pretrained models.

    This function replicates the original ``cascade2p.cascade.predict()`` logic
    exactly. It loads pretrained ensemble models, matches neurons to
    noise-level-specific models, runs batched inference, and applies
    thresholding.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. ``'Universal_30Hz_smoothing100ms'``.
        Must correspond to a folder containing ``config.yaml`` and ``.pth`` files.
    traces : numpy.ndarray
        dF/F traces of shape ``(neurons, nr_timepoints)`` (fractions, not percent).
    model_folder : str
        Path to the directory containing the model folder.
    threshold : int or bool
        - ``0``: Set all negative values to 0.
        - ``1`` or ``True``: Threshold to remove signals smaller than 1/e of
          a single action potential (with dilated mask).
        - ``False``: No thresholding; result can contain negative values.
    padding : float
        Value inserted where no prediction can be made (window edges).
        Default is ``np.nan``; ``0`` is also common.
    trace_noise_levels : numpy.ndarray, optional
        Pre-computed noise levels. If ``None``, they are calculated from traces.
    verbosity : int
        If 0, suppress console output during inference.
    device : torch.device, optional
        Device for inference. If ``None``, auto-selects CUDA if available.

    Returns
    -------
    numpy.ndarray
        Predicted spiking activity of shape ``(neurons, nr_timepoints)``.
        May contain NaN values at the edges if ``padding=np.nan``.
    """
    from cascade2p import config
    from cascade2p.utils import preprocess_traces

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reshape if only a single neuron is provided
    if len(traces.shape) == 1:
        traces = np.expand_dims(traces, 0)

    model_path = os.path.join(model_folder, model_name)
    cfg_file = os.path.join(model_path, "config.yaml")

    # check if configuration file can be found
    if not os.path.isfile(cfg_file):
        m = (
            'The configuration file "config.yaml" can not be found at the '
            'location "{}".\n'.format(os.path.abspath(cfg_file))
            + 'You have provided the model "{}" at the absolute or relative '
            'path "{}".\n'.format(model_name, model_folder)
            + 'Please check if there is a folder for model "{}" at the '
            'location "{}".'.format(model_name, os.path.abspath(model_folder))
        )
        print(m)
        raise Exception(m)

    # Load config file
    cfg = config.read_config(cfg_file)

    # extract values from config file
    verbose = cfg["verbose"]
    if verbosity == 0:
        verbose = 0
    training_data = cfg["training_datasets"]
    ensemble_size = cfg["ensemble_size"]
    batch_size = cfg["batch_size"]
    sampling_rate = cfg["sampling_rate"]
    before_frac = cfg["before_frac"]
    window_size = cfg["windowsize"]
    noise_levels_model = cfg["noise_levels"]
    smoothing = cfg["smoothing"]
    causal_kernel = cfg["causal_kernel"]

    model_description = (
        "\n \nThe selected model was trained on "
        + str(len(training_data))
        + " datasets, with "
        + str(ensemble_size)
        + " ensembles for each noise level, at a sampling rate of "
        + str(sampling_rate)
        + "Hz,"
    )
    if causal_kernel:
        model_description += (
            " with a resampled ground truth that was smoothed with a causal kernel"
        )
    else:
        model_description += (
            " with a resampled ground truth that was smoothed with a Gaussian kernel"
        )
    model_description += (
        " of a standard deviation of "
        + str(int(1000 * smoothing))
        + " milliseconds. \n \n"
    )
    if verbose:
        print(model_description)

    if verbose:
        print("Loaded model was trained at frame rate {} Hz".format(sampling_rate))
    if verbose:
        print(
            "Given argument traces contains {} neurons and {} frames.".format(
                traces.shape[0], traces.shape[1]
            )
        )

    if trace_noise_levels is None:
        # calculate noise levels for each trace
        trace_noise_levels = calculate_noise_levels(traces, sampling_rate)

    if verbose:
        print(
            "Noise levels (mean, std; in standard units): "
            + str(int(np.nanmean(trace_noise_levels * 100)) / 100)
            + ", "
            + str(int(np.nanstd(trace_noise_levels * 100)) / 100)
        )

    # Get model paths as dictionary (key: noise_level)
    model_dict = get_model_paths(model_path)
    if verbose > 2:
        print("Loaded models:", str(model_dict))

    # XX has shape: (neurons, timepoints, windowsize)
    XX = preprocess_traces(traces, before_frac=before_frac, window_size=window_size)
    Y_predict = np.zeros((XX.shape[0], XX.shape[1]))

    # Compute difference of noise levels between each neuron and each model
    differences = (
        np.array(trace_noise_levels)[:, None]
        - np.array(noise_levels_model)[None, :]
    )
    relative_differences = np.min(differences, axis=1)
    if np.mean(relative_differences) > 2:
        print(
            "WARNING: The available models cannot match the experimentally "
            "obtained noise levels (difference: ",
            str(np.mean(relative_differences)),
            "). Please check that the computation of dF/F is performed correctly. "
            "Otherwise, please reach out and ask for pretrained models with higher "
            "noise level models (see: "
            "https://github.com/HelmchenLabSoftware/Cascade/issues/61).",
        )
    best_model_for_each_neuron = np.argmin(np.abs(differences), axis=1)

    # Use for each noise level the matching model
    for i, model_noise in enumerate(noise_levels_model):

        if verbose:
            print("\nPredictions for noise level {}:".format(model_noise))

        # select neurons which have this noise level
        neuron_idx = np.where(best_model_for_each_neuron == i)[0]

        if len(neuron_idx) == 0:
            if verbose:
                print("\tNo neurons for this noise level")
            continue

        # load PyTorch models for the given noise level
        models = []
        for model_path_file in model_dict[model_noise]:
            net = CascadeNet(
                filter_sizes=cfg["filter_sizes"],
                filter_numbers=cfg["filter_numbers"],
                dense_expansion=cfg["dense_expansion"],
                windowsize=cfg["windowsize"],
            )
            net.load_state_dict(
                torch.load(model_path_file, map_location=device, weights_only=True)
            )
            net.to(device)
            net.eval()
            models.append(net)

        # select neurons and merge neurons and timepoints into one dimension
        XX_sel = XX[neuron_idx, :, :]

        XX_sel = np.reshape(
            XX_sel, (XX_sel.shape[0] * XX_sel.shape[1], XX_sel.shape[2])
        )
        XX_sel = np.expand_dims(
            XX_sel, axis=2
        )  # add empty third dimension to match training shape

        for j, model in enumerate(models):
            if verbose:
                print("\t... ensemble", j)

            # Convert to PyTorch tensor
            XX_tensor = torch.FloatTensor(XX_sel).to(device)

            # Create DataLoader for batched inference
            dataset = torch.utils.data.TensorDataset(XX_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False
            )

            # Perform inference
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    batch_X = batch[0]
                    outputs = model(batch_X)
                    predictions.append(outputs.cpu().numpy())

            prediction_flat = np.concatenate(predictions, axis=0)
            prediction = np.reshape(
                prediction_flat, (len(neuron_idx), XX.shape[1])
            )

            Y_predict[neuron_idx, :] += prediction / len(models)

        # Clean up models from memory
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if threshold is False:  # only if 'False' is passed as argument
        if verbose:
            print(
                "Skipping the thresholding. There can be negative values in the result."
            )

    elif threshold == 1:  # (1 or True)
        # Cut off noise floor (lower than 1/e of a single action potential)
        from scipy.ndimage import gaussian_filter, binary_dilation

        # find out empirically how large a single AP is
        single_spike = np.zeros(1001)
        single_spike[501] = 1
        single_spike_smoothed = gaussian_filter(
            single_spike.astype(float), sigma=smoothing * sampling_rate
        )
        threshold_value = np.max(single_spike_smoothed) / np.exp(1)

        # Set everything below threshold to zero.
        # Use binary dilation to avoid clipping of true events.
        for neuron in range(Y_predict.shape[0]):
            with np.errstate(invalid="ignore"):
                activity_mask = Y_predict[neuron, :] > threshold_value
            activity_mask = binary_dilation(
                activity_mask, iterations=int(smoothing * sampling_rate)
            )

            Y_predict[neuron, ~activity_mask] = 0

            Y_predict[Y_predict < 0] = 0

    elif threshold == 0:
        with np.errstate(invalid="ignore"):
            Y_predict[Y_predict < 0] = 0

    else:
        raise Exception(
            'Invalid value of threshold "{}". Only 0, 1 (or True) or False allowed'.format(
                threshold
            )
        )

    # NaN or 0 for first and last datapoints where no predictions can be made
    Y_predict[:, 0 : int(before_frac * window_size)] = padding
    Y_predict[:, -int((1 - before_frac) * window_size) :] = padding

    print("Spike rate inference done.")

    return Y_predict
