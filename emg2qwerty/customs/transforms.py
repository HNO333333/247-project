import neurokit2 as nk
import torch
import torchaudio
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Sequence
import warnings
from einops import rearrange
warnings.filterwarnings("ignore", category=nk.NeuroKitWarning)


@dataclass
class NKClean:
    """
    NKClean is a data transformation class that processes EMG signals using NeuroKit2.

    This class applies the `nk.emg_process` function to clean the EMG data for each specified field
    and channel in the input structured numpy array. The cleaned data is then returned in the same
    format as the input.

    Attributes:
        sampling_rate (int): The sampling rate used for processing the EMG signals. Default is 2000 Hz.
        fields (Sequence[str]): A sequence of field names to be processed from the input data. 
                                Default fields are ("emg_left", "emg_right").

    Methods:
        __call__(data: np.ndarray) -> np.ndarray:
            Processes the input data and returns the cleaned EMG signals.
    """
    sampling_rate: int = 2000
    fields: Sequence[str] = ("emg_left", "emg_right")
    def __call__(self, data: np.ndarray) -> np.ndarray:
        for field in self.fields:
            for channel in range(data[field].shape[1]):
                data[field][:, channel] = nk.emg_clean(data[field][:, channel], sampling_rate=self.sampling_rate)
        return data


@dataclass
class CombineChannels:
    """
    combine num_bands * num_channels into a single channel, and unsqueeze the conv channel dimension

    in: (time, num_bands, num_channels)
    out: (1, channels, time), channels = num_bands * num_channels
    """
    def __call__(self, data: torch.tensor) -> torch.tensor:
        # input shape: (time, num_bands, num_channels)
        # output shape: (time, 1, channels), channels = num_bands * num_channels
        return rearrange(data, "t bd ec -> t (bd ec)").unsqueeze(1)

@dataclass
class UnsqueezeConvChannel:
    def __call__(self, data: torch.tensor) -> torch.tensor:
        # input shape: (time, num_bands, num_channels)
        # output shape: (time, 1, num_bands, num_channels)
        return data.unsqueeze(1)


@dataclass
class CropFreqBins:
    start_bin_idx: int = 1
    end_bin_idx: int = 17
    def __call__(self, data: torch.tensor) -> torch.tensor:
        # input shape: (time, num_bands, num_channels, freq)
        # output shape: (time, num_bands, num_channels, freq - start_bin_idx - end_bin_idx)
        return data[:, :, :, self.start_bin_idx:self.end_bin_idx]

# NOTE: cropping freq bins is kind of rude... downsample first then take spectrogram might be better
@dataclass
class Downsample:
    factor: int = 2
    sr: int = 2000
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = data.movedim(0, -1) # move time to last dim
        data = torchaudio.functional.resample(data, orig_freq=self.sr, new_freq=self.sr // self.factor)
        return data.movedim(-1, 0) # move time back to first dim