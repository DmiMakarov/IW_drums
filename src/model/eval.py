from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import typer

from src.model import OnsetLightningModule, SincNetDSN

app = typer.Typer(help="Evaluate a .m4a file and produce onsets.txt")


def _load_audio_mono(path: Path, target_sr: int) -> torch.Tensor:
    """Load audio as mono float32 at target sample rate. Returns shape (1, T)."""
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        # Fallback to ffmpeg decoding if torchaudio backend can't handle .m4a
        ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not ffmpeg_path:
            raise
        cmd = [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode(errors="ignore")
            raise RuntimeError(f"ffmpeg failed to decode {path.name}: {err}")
        import numpy as np

        audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        wav = torch.from_numpy(audio).unsqueeze(0)
        sr = target_sr
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(torch.float32).contiguous()


def _pick_onsets_from_probs(
    probs: torch.Tensor,
    hop_size: int,
    sample_rate: int,
    threshold: float,
    min_separation_s: float,
) -> list[float]:
    """Simple peak picking: choose rising edges over threshold, spaced apart."""
    p = probs.detach().cpu().flatten()
    if p.numel() == 0:
        return []

    # Rising edge detection
    above = p >= threshold
    prev = torch.zeros_like(above)
    prev[1:] = above[:-1]
    rising = above & (~prev)

    # Convert frame indices to times
    onset_frames = torch.nonzero(rising, as_tuple=False).flatten().tolist()
    onset_times: list[float] = []
    min_sep_samples = int(round(min_separation_s * sample_rate))
    last_sample_time = -10_000_000
    for f in onset_frames:
        sample_idx = int(f) * hop_size
        if sample_idx - last_sample_time < min_sep_samples:
            continue
        onset_times.append(sample_idx / sample_rate)
        last_sample_time = sample_idx
    return onset_times


def _latest_checkpoint(default_dir: Path) -> Optional[Path]:
    if not default_dir.exists():
        return None
    candidates = sorted(
        default_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


@app.command()
def main(
    audio_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, resolve_path=True, help="Input .m4a file"
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to a Lightning .ckpt file. Defaults to latest in ./checkpoints",
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Output onsets file. Defaults to <audio_stem>_onsets.txt alongside input.",
    ),
    frame_size: int = typer.Option(2048, min=1, help="Frame size in samples"),
    hop_size: int = typer.Option(512, min=1, help="Hop size in samples"),
    sample_rate: int = typer.Option(44100, min=1, help="Target sample rate"),
    threshold: float = typer.Option(
        0.5, min=0.0, max=1.0, help="Probability threshold for onset"
    ),
    batch_size: int = typer.Option(2048, min=1, help="Batch size for inference"),
    min_separation_s: float = typer.Option(
        0.03, min=0.0, help="Minimum separation between onsets (s)"
    ),
    device: str = typer.Option("auto", help="Device: 'auto', 'cpu', or 'cuda'"),
):
    """Run inference and write detected onset timestamps to a text file."""

    # Resolve checkpoint
    if checkpoint is None:
        checkpoint = _latest_checkpoint(Path("checkpoints"))
        if checkpoint is None:
            raise typer.BadParameter(
                "No checkpoint provided and none found in ./checkpoints"
            )

    # Resolve output path
    if output is None:
        output = audio_path.with_name(f"{audio_path.stem}_onsets.txt")

    # Select device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # Load model
    backbone = SincNetDSN()
    model = OnsetLightningModule.load_from_checkpoint(
        str(checkpoint), backbone=backbone, map_location=dev
    )
    model.eval()
    model.to(dev)

    # Load audio and build frames via unfold
    wav = _load_audio_mono(audio_path, sample_rate)
    T = wav.size(1)
    if T < frame_size:
        typer.echo("Audio shorter than frame_size; no onsets will be detected.")
        with open(output, "w", encoding="utf-8") as f:
            pass
        return

    n_frames = 1 + (T - frame_size) // hop_size
    frames = wav.unfold(
        dimension=1, size=frame_size, step=hop_size
    )  # (1, n_frames, frame)
    frames = frames.squeeze(0)  # (n_frames, frame)
    frames = frames.unsqueeze(1)  # (n_frames, 1, frame)

    # Batched inference
    probs_list = []
    with torch.no_grad():
        for i in range(0, frames.size(0), batch_size):
            batch = frames[i : i + batch_size].to(dev)
            logits = model.model(batch)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.detach().cpu())
    probs_all = torch.cat(probs_list, dim=0)

    # Pick onsets
    onset_times = _pick_onsets_from_probs(
        probs_all,
        hop_size=hop_size,
        sample_rate=sample_rate,
        threshold=threshold,
        min_separation_s=min_separation_s,
    )

    # Write output
    with open(output, "w", encoding="utf-8") as f:
        for t in onset_times:
            f.write(f"{t:.6f}\n")

    typer.echo(f"Wrote {len(onset_times)} onsets to {output}")


if __name__ == "__main__":
    app()
