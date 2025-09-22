from __future__ import annotations

from pathlib import Path
from typing import Optional

import queue
import threading
import time

import numpy as np
import torch
import typer

from model import SincNetDSN, OnsetLightningModule


app = typer.Typer(help="Real-time onset detection from microphone input")


def _latest_checkpoint(default_dir: Path) -> Optional[Path]:
    if not default_dir.exists():
        return None
    candidates = sorted(
        default_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


class RTOnsetDetector:
    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate: int,
        frame_size: int,
        hop_size: int,
        threshold: float,
        min_separation_s: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.sr = sample_rate
        self.frame = frame_size
        self.hop = hop_size
        self.thresh = threshold
        self.min_sep = int(round(min_separation_s * sample_rate))
        self.device = device

        self.window = np.zeros(self.frame, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.frame_index = 0
        self.prev_above = False
        self.last_onset_sample = -10_000_000

        self._lock = threading.Lock()

    def process_samples(self, samples: np.ndarray) -> list[tuple[float, float]]:
        """
        Push raw mono float32 samples; returns list of detected onsets
        as (time_seconds, probability) tuples.
        """
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)

        with self._lock:
            if self.pending.size == 0:
                self.pending = samples.copy()
            else:
                self.pending = np.concatenate([self.pending, samples])

            onsets: list[tuple[float, float]] = []
            # Consume in hop-sized chunks
            while self.pending.size >= self.hop:
                hop_chunk = self.pending[: self.hop]
                self.pending = self.pending[self.hop :]

                # Update sliding window
                self.window[:-self.hop] = self.window[self.hop :]
                self.window[-self.hop :] = hop_chunk

                # Inference
                with torch.no_grad():
                    x = torch.from_numpy(self.window).to(self.device)
                    x = x.view(1, 1, -1)
                    logits = self.model(x)
                    prob = torch.sigmoid(logits).item()

                # Rising edge detection with min separation
                above = prob >= self.thresh
                if above and not self.prev_above:
                    sample_idx = self.frame_index * self.hop
                    if sample_idx - self.last_onset_sample >= self.min_sep:
                        t = sample_idx / self.sr
                        onsets.append((t, float(prob)))
                        self.last_onset_sample = sample_idx
                self.prev_above = above
                self.frame_index += 1

            return onsets


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@app.command()
def main(
    checkpoint: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to a Lightning .ckpt file. Defaults to latest in ./checkpoints",
    ),
    sample_rate: int = typer.Option(44100, min=1, help="Stream sample rate"),
    frame_size: int = typer.Option(2048, min=1, help="Frame size in samples"),
    hop_size: int = typer.Option(512, min=1, help="Hop size in samples"),
    threshold: float = typer.Option(
        0.5, min=0.0, max=1.0, help="Probability threshold for onset"
    ),
    min_separation_s: float = typer.Option(
        0.03, min=0.0, help="Minimum separation between onsets (s)"
    ),
    device: str = typer.Option("auto", help="Model device: 'auto', 'cpu', or 'cuda'"),
    input_device: Optional[str] = typer.Option(
        None,
        help="sounddevice input device (index or name). Defaults to system default.",
    ),
    blocksize: int = typer.Option(
        0,
        min=0,
        help="sounddevice blocksize (0 lets backend choose). Typically set to hop_size for low latency.",
    ),
    show_prob: bool = typer.Option(
        False, help="Print continuous probabilities (can be verbose)."
    ),
):
    """Capture microphone audio and print onset times in real time."""

    # Resolve checkpoint
    if checkpoint is None:
        checkpoint = _latest_checkpoint(Path("checkpoints"))
        if checkpoint is None:
            raise typer.BadParameter(
                "No checkpoint provided and none found in ./checkpoints"
            )

    dev = _resolve_device(device)

    # Load model
    backbone = SincNetDSN(sr=sample_rate)
    pl_module = OnsetLightningModule.load_from_checkpoint(
        str(checkpoint), backbone=backbone, map_location=dev
    )
    pl_module.eval()
    pl_module.to(dev)
    model = pl_module.model  # use backbone for forward
    model.eval()
    model.to(dev)

    detector = RTOnsetDetector(
        model=model,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        threshold=threshold,
        min_separation_s=min_separation_s,
        device=dev,
    )

    # Audio input via sounddevice
    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover
        typer.echo(
            "sounddevice is required for realtime mic input. Install with 'pip install sounddevice'.",
            err=True,
        )
        raise e

    q: queue.Queue[np.ndarray] = queue.Queue()
    stop_flag = threading.Event()

    def audio_callback(indata, frames, time_info, status):  # type: ignore[no-redef]
        if status:
            # Print at most once per block
            print(f"[audio] {status}")
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            pass

    stream_kwargs = dict(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=(hop_size if blocksize == 0 else blocksize),
        callback=audio_callback,
    )
    if input_device is not None:
        # Typer doesn't support Union[int,str]; accept string and coerce if numeric
        try:
            stream_kwargs["device"] = int(input_device)
        except ValueError:
            stream_kwargs["device"] = input_device

    typer.echo(
        f"Listening… SR={sample_rate} frame={frame_size} hop={hop_size} threshold={threshold}"
    )
    typer.echo("Press Ctrl+C to stop.")

    # Consumer loop
    t0 = time.perf_counter()

    try:
        with sd.InputStream(**stream_kwargs):  # type: ignore[name-defined]
            while not stop_flag.is_set():
                try:
                    data = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                onsets = detector.process_samples(data[:, 0] if data.ndim == 2 else data)

                if show_prob:
                    # Compute last prob again for display (already computed inside detector,
                    # but we keep this simple: display only on onset or every processed hop is too noisy)
                    pass  # keep output minimal unless onsets fire

                for t, p in onsets:
                    print(f"Onset @ {t:.3f}s  p={p:.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()


if __name__ == "__main__":
    app()
