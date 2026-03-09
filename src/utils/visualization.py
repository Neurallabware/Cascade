"""Visualization utilities for CASCADE calcium imaging analysis.

Extracted from cascade2p/utils.py -- scientific logic preserved exactly.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .noise import calculate_noise_levels


def plot_dFF_traces(
    traces: NDArray[np.floating],
    neuron_indices: list[int] | NDArray[np.integer],
    frame_rate: float,
    spiking: NDArray[np.floating] | None = None,
    discrete_spikes: list | None = None,
    y_range: tuple[float, float] = (-1.5, 2),
) -> NDArray[np.floating]:
    """Plot a subset (max 50 seconds) of calcium imaging dF/F traces.

    Parameters
    ----------
    traces : numpy.ndarray
        dF/F traces with shape ``(nb_neurons, time_points)``.
    neuron_indices : list or array of int
        Indices of neurons to plot.
    frame_rate : float
        Sampling rate in Hz.
    spiking : numpy.ndarray, optional
        Spike prediction data with the same shape as *traces*.
    discrete_spikes : list, optional
        List of spike time arrays (one per neuron).
    y_range : tuple of float
        Y-axis limits for the plots.

    Returns
    -------
    numpy.ndarray
        Time vector used for the x-axis.
    """
    try:
        import seaborn as sns
        sns.set()
        plt.style.use("seaborn-darkgrid")
    except Exception:
        pass

    t_max = int(np.minimum(50.0, traces.shape[1] / frame_rate) * frame_rate)
    traces = traces[:, :t_max]

    time = np.arange(0, traces.shape[1]) / frame_rate

    fig, axs = plt.subplots(
        int(np.ceil(len(neuron_indices) / 2)), 2, sharex=True, sharey=True
    )
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    plt.xlabel("Time (s)")
    plt.ylabel("dF/F (values normally between 0 and 4)")

    for k, neuron_index in enumerate(neuron_indices):

        subplot_ix = int(k / 2), int(np.mod(k, 2))
        axs[subplot_ix].plot(time, traces[neuron_index, :])
        axs[subplot_ix].set_ylim(y_range)
        axs[subplot_ix].set_xlim(
            32 / frame_rate, t_max / frame_rate - 32 / frame_rate
        )

        if spiking is not None:
            axs[subplot_ix].plot(time, spiking[neuron_index, :t_max] - 1)

        if discrete_spikes is not None:
            for spike_time in discrete_spikes[neuron_index]:
                axs[subplot_ix].plot(
                    np.array([spike_time, spike_time]) / frame_rate + 1 / frame_rate,
                    [-1.4, -1.2],
                    "k",
                )

    return time


def plot_noise_level_distribution(
    traces: NDArray[np.floating],
    frame_rate: float,
) -> NDArray[np.floating]:
    """Plot a histogram of noise levels across all neurons.

    Parameters
    ----------
    traces : numpy.ndarray
        dF/F traces with shape ``(nb_neurons, time_points)``.
    frame_rate : float
        Sampling rate in Hz.

    Returns
    -------
    numpy.ndarray
        Noise levels for all neurons.
    """
    try:
        import seaborn as sns
        sns.set()
        plt.style.use("seaborn-darkgrid")
    except Exception:
        pass

    noise_levels = calculate_noise_levels(traces, frame_rate)

    percent999 = np.nanpercentile(noise_levels, 99.9)

    plt.figure(1121)
    plt.hist(noise_levels, density=True, bins=100)
    plt.xlim([0, percent999])
    plt.xlabel("Noise level (% s^(1/2))")
    plt.title("Histogram of noise levels across neurons")

    return noise_levels


def plot_noise_matched_ground_truth(
    model_name: str,
    median_noise: float,
    frame_rate: float,
    nb_traces: int,
    duration: float,
) -> None:
    """Plot calcium traces alongside electrophysiological ground truth.

    Plots subsets (chunks of ``duration`` seconds) of calcium imaging data
    together with ground truth from the datasets used for training the model.

    Parameters
    ----------
    model_name : str
        Name of the pretrained model.
    median_noise : float
        Median noise level of the dataset.
    frame_rate : float
        Sampling rate in Hz.
    nb_traces : int
        Number of example traces to plot.
    duration : float
        Duration in seconds for each plotted trace.
    """
    # Import from the original cascade2p package for ground truth processing
    from cascade2p import config
    from cascade2p.utils import preprocess_groundtruth_artificial_noise_balanced

    model_folder = os.path.join("Pretrained_models", model_name)

    # Load config file
    cfg = config.read_config(os.path.join(model_folder, "config.yaml"))

    # extract values from config file into variables
    training_folders = [
        os.path.join("Ground_truth", ds) for ds in cfg["training_datasets"]
    ]

    # rename training folder names for models trained before nomenclature change (2021)
    if (
        "Ground_truth/DS08-GCaMP6f-m-V1" in training_folders
        or "Ground_truth/DS03-OGB1-zf-pDp" in training_folders
    ):
        _renames = [
            ("Ground_truth/DS02-Cal520-m-S1", "Ground_truth/DS03-Cal520-m-S1"),
            ("Ground_truth/DS03-OGB1-zf-pDp", "Ground_truth/DS04-OGB1-zf-pDp"),
            ("Ground_truth/DS04-Cal520-zf-pDp", "Ground_truth/DS05-Cal520-zf-pDp"),
            ("Ground_truth/DS05-GCaMP6f-zf-aDp", "Ground_truth/DS06-GCaMP6f-zf-aDp"),
            ("Ground_truth/DS06-GCaMP6f-zf-dD", "Ground_truth/DS07-GCaMP6f-zf-dD"),
            ("Ground_truth/DS07-GCaMP6f-zf-OB", "Ground_truth/DS08-GCaMP6f-zf-OB"),
            ("Ground_truth/DS08-GCaMP6f-m-V1", "Ground_truth/DS09-GCaMP6f-m-V1"),
            ("Ground_truth/DS15-GCaMP6s-m-V1", "Ground_truth/DS16-GCaMP6s-m-V1"),
            ("Ground_truth/DS14-GCaMP6s-m-V1", "Ground_truth/DS15-GCaMP6s-m-V1"),
            ("Ground_truth/DS13-GCaMP6s-m-V1", "Ground_truth/DS14-GCaMP6s-m-V1"),
            ("Ground_truth/DS10-GCaMP6f-m-V1-neuropil-corrected", "Ground_truth/DS11-GCaMP6f-m-V1-neuropil-corrected"),
            ("Ground_truth/DS09-GCaMP6f-m-V1-neuropil-corrected", "Ground_truth/DS10-GCaMP6f-m-V1-neuropil-corrected"),
            ("Ground_truth/DS12-GCaMP6s-m-V1-neuropil-corrected", "Ground_truth/DS13-GCaMP6s-m-V1-neuropil-corrected"),
            ("Ground_truth/DS11-GCaMP6s-m-V1-neuropil-corrected", "Ground_truth/DS12-GCaMP6s-m-V1-neuropil-corrected"),
            ("Ground_truth/DS16-GCaMP5k-m-V1", "Ground_truth/DS17-GCaMP5k-m-V1"),
            ("Ground_truth/DS17-R-CaMP-m-CA3", "Ground_truth/DS18-R-CaMP-m-CA3"),
            ("Ground_truth/DS18-R-CaMP-m-S1", "Ground_truth/DS19-R-CaMP-m-S1"),
            ("Ground_truth/DS19-jRCaMP1a-m-V1", "Ground_truth/DS20-jRCaMP1a-m-V1"),
        ]
        for old, new in _renames:
            training_folders = [w.replace(old, new) for w in training_folders]

    # extract ground truth
    X, Y = preprocess_groundtruth_artificial_noise_balanced(
        ground_truth_folders=training_folders,
        before_frac=cfg["before_frac"],
        windowsize=cfg["windowsize"],
        after_frac=1 - cfg["before_frac"],
        noise_level=median_noise,
        sampling_rate=cfg["sampling_rate"],
        smoothing=cfg["smoothing"] * cfg["sampling_rate"],
        omission_list=[],
        permute=0,
        verbose=cfg["verbose"],
        replicas=0,
    )

    X = X[:, int(cfg["windowsize"] / 2), ]
    Y = Y[:, ]

    # Plotting (very similar to plot_dFF_traces)
    try:
        import seaborn as sns
        sns.set()
        plt.style.use("seaborn-darkgrid")
    except Exception:
        pass

    duration_datapoints = int(duration * frame_rate)

    time_indices = np.random.randint(X.shape[0] - duration_datapoints, size=nb_traces)

    time = np.arange(0, int(duration * frame_rate)) / frame_rate

    fig, axs = plt.subplots(
        int(np.ceil(nb_traces / 2)), 2, sharex=True, sharey=True
    )
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    plt.xlabel("Time (s)")
    plt.ylabel("dF/F and ground truth spiking")

    for k, time_index in enumerate(time_indices):

        subplot_ix = int(k / 2), int(np.mod(k, 2))
        axs[subplot_ix].plot(
            time, X[time_index : time_index + duration_datapoints, :]
        )
        axs[subplot_ix].set_xlim(0, duration)

        axs[subplot_ix].plot(
            time, Y[time_index : time_index + duration_datapoints] - 1
        )
