"""CascadeNet: 1D CNN for spike inference from calcium imaging data.

Pure nn.Module extracted from the original define_model() in cascade2p/utils.py.
Architecture: 3 Conv1d layers, 2 MaxPool1d layers, 1 per-timestep dense layer,
and a final regression head producing a single scalar output.
"""

from __future__ import annotations

import torch.nn as nn


class CascadeNet(nn.Module):
    """1D convolutional neural network for calcium-to-spike inference.

    The network takes a windowed dF/F trace of shape ``(batch, windowsize, 1)``
    and produces a single scalar spike-rate prediction per sample.

    Architecture (identical to the original Cascade model):
        Conv1d -> ReLU -> Conv1d -> ReLU -> MaxPool1d(2) ->
        Conv1d -> ReLU -> MaxPool1d(2) ->
        Linear (per-timestep dense) -> ReLU -> Flatten -> Linear -> output

    Parameters
    ----------
    filter_sizes : list[int]
        Kernel sizes for the three Conv1d layers.
    filter_numbers : list[int]
        Number of output channels for the three Conv1d layers.
    dense_expansion : int
        Number of units in the per-timestep dense layer.
    windowsize : int
        Length of the input time window (number of timepoints).
    """

    def __init__(
        self,
        filter_sizes: list[int] | None = None,
        filter_numbers: list[int] | None = None,
        dense_expansion: int = 10,
        windowsize: int = 64,
    ) -> None:
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [31, 19, 5]
        if filter_numbers is None:
            filter_numbers = [30, 40, 50]

        self.filter_sizes = list(filter_sizes)
        self.filter_numbers = list(filter_numbers)
        self.dense_expansion = dense_expansion
        self.windowsize = windowsize

        flattened_size = self._calculate_flattened_size(windowsize, self.filter_sizes)
        if flattened_size <= 0:
            min_windowsize = self._minimum_supported_windowsize(self.filter_sizes)
            raise ValueError(
                "windowsize={} is too small for filter_sizes={}. "
                "The default CascadeNet architecture requires windowsize >= {}."
                .format(windowsize, self.filter_sizes, min_windowsize)
            )

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, filter_numbers[0], filter_sizes[0], stride=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(filter_numbers[0], filter_numbers[1], filter_sizes[1])
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(filter_numbers[1], filter_numbers[2], filter_sizes[2])
        self.relu3 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(2)

        # Per-timestep dense layer (Linear applied over the channel dimension)
        self.dense1 = nn.Linear(filter_numbers[2], dense_expansion)
        self.relu4 = nn.ReLU()

        # Final regression head
        flattened_size = flattened_size * dense_expansion
        self.dense2 = nn.Linear(flattened_size, 1)

    @staticmethod
    def _calculate_flattened_size(windowsize: int, filter_sizes: list[int]) -> int:
        """Compute the temporal sequence length after convolutions and pooling."""
        size = windowsize
        size = size - (filter_sizes[0] - 1)  # conv1
        size = size - (filter_sizes[1] - 1)  # conv2
        size = size // 2                      # pool1
        size = size - (filter_sizes[2] - 1)  # conv3
        size = size // 2                      # pool2
        return size

    @staticmethod
    def _minimum_supported_windowsize(filter_sizes: list[int]) -> int:
        """Return the smallest window size that yields a positive flattened size."""
        candidate = 1
        while CascadeNet._calculate_flattened_size(candidate, filter_sizes) <= 0:
            candidate += 1
        return candidate

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, windowsize, 1)``.

        Returns
        -------
        torch.Tensor
            Predicted spike rate of shape ``(batch, 1)``.
        """
        # (batch, windowsize, 1) -> (batch, 1, windowsize) for Conv1d
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.pool2(x)

        # (batch, channels, time) -> (batch, time, channels) for per-timestep dense
        x = x.permute(0, 2, 1)

        x = self.dense1(x)
        x = self.relu4(x)

        # Flatten all timesteps * channels
        x = x.reshape(x.size(0), -1)

        x = self.dense2(x)

        return x
