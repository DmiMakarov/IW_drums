from __future__ import annotations

import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryF1Score


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=50,
        min_band_hz=50,
    ):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # depthwise
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class SincNetDSN(nn.Module):
    """
    Hybrid: Sinc-based front-end + depthwise separable CNN head.
    Input: (batch, 1, T)
    Output: (batch,) logits
    """

    def __init__(
        self,
        sr: int = 44100,
        sinc_channels: int = 24,
        sinc_kernel: int = 251,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sinc = SincConv_fast(
            out_channels=sinc_channels, kernel_size=sinc_kernel, sample_rate=sr
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(sinc_channels),
            nn.ReLU(),
            DepthwiseSeparableConv1d(
                sinc_channels, 16, kernel_size=5, stride=2, padding=2
            ),
            nn.MaxPool1d(2),
            DepthwiseSeparableConv1d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveMaxPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sinc(x)
        x = self.head(x)
        return self.fc(x).squeeze(-1)


class OnsetLightningModule(pl.LightningModule):
    """Lightning wrapper that handles optimisation and metrics."""

    def __init__(
        self,
        backbone: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.model = backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_f1 = BinaryF1Score(threshold=threshold)
        self.val_f1 = BinaryF1Score(threshold=threshold)

        # Metrics are updated per step; computed and reset at epoch end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        y = y.view(-1).to(torch.float32)
        logits = self.model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        probs = torch.sigmoid(logits)
        return loss, probs, y

    def training_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        self.train_f1.update(probs, targets.int())
        self.log(
            "train/loss", loss, prog_bar=True, on_epoch=True, batch_size=targets.size(0)
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        self.val_f1.update(probs, targets.int())
        self.log(
            "val/loss", loss, prog_bar=True, on_epoch=True, batch_size=targets.size(0)
        )
        return loss

    def on_train_epoch_end(self):
        try:
            f1 = self.train_f1.compute()
            if f1 is not None:
                self.log("train/f1", f1, prog_bar=False)
        finally:
            self.train_f1.reset()

    def on_validation_epoch_end(self):
        try:
            f1 = self.val_f1.compute()
            if f1 is not None:
                self.log("val/f1", f1, prog_bar=True)
        finally:
            self.val_f1.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


# Convenience helpers


def frame_to_tensor(frame, device=None):
    """Convert a mono 1D array to tensor shaped ``(1, 1, T)``."""
    import numpy as np

    frame = np.asarray(frame).astype("float32")
    t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
    return t.to(device) if device is not None else t


def predict_onset_probability(model, frame, device=None):
    """Return the onset probability for a single frame."""
    model.eval()
    if isinstance(frame, torch.Tensor):
        x = frame
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
    else:
        x = frame_to_tensor(frame, device)
    if device is not None:
        model = model.to(device)
        x = x.to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
    return float(probs.squeeze().detach().cpu().item())
