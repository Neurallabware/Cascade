"""Noise level estimation for calcium imaging traces.

Extracted from cascade2p/utils.py -- scientific logic preserved exactly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def calculate_noise_levels(
    neurons_x_time: NDArray[np.floating],
    frame_rate: float,
) -> NDArray[np.floating]:
    """Compute the noise level for each neuron from a dF/F matrix.

    The noise level is computed as the median absolute dF/F difference
    between two subsequent time points. This is an outlier-robust measurement
    that converges to the simple standard deviation of the dF/F trace for
    uncorrelated and outlier-free dF/F traces.

    The value is divided by the square root of the frame rate in order to make
    it comparable across recordings with different frame rates.

    Parameters
    ----------
    neurons_x_time : numpy.ndarray
        Matrix with shape ``(nb_neurons, time_points)`` containing dF/F traces.
    frame_rate : float
        Sampling rate of the calcium recording in Hz.

    Returns
    -------
    numpy.ndarray
        Vector of noise levels (in percent) for all neurons.
    """
    noise_levels = (
        np.nanmedian(np.abs(np.diff(neurons_x_time, axis=-1)), axis=-1)
        / np.sqrt(frame_rate)
    )
    return noise_levels * 100  # scale noise levels to percent
