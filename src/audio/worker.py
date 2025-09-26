"""Background worker that processes audio input and detects drumstick hits."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd
import torch

from src.model.model import SincNetDSN, OnsetLightningModule

logger = logging.getLogger(__name__)


class AudioWorker:
    """Background worker that processes audio input and detects drumstick hits.

    Uses the trained model to detect drumstick hits and provides callbacks
    for music control based on hit patterns.
    """

    def __init__(
        self,
        on_start_music: Callable[[], None],
        on_stop_music: Callable[[], None],
        on_status_change: Callable[[bool], None] | None = None,
        checkpoint_path: str = "src/model/onset-detector.ckpt",
        sample_rate: int = 44100,
        frame_size: int = 2048,
        hop_size: int = 512,
        threshold: float = 0.9,
        min_separation_s: float = 0.3,
        device: str = "auto",
        input_device: str | None = None,
    ) -> None:
        """Initialize the audio worker."""
        self.on_start_music = on_start_music
        self.on_stop_music = on_stop_music
        self.on_status_change = on_status_change
        self.checkpoint_path = checkpoint_path
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.threshold = threshold
        self.min_separation_s = min_separation_s
        self.device = self._resolve_device(device)
        self.input_device = input_device

        self._thread: threading.Thread | None = None
        self._running = False
        self._model: torch.nn.Module | None = None
        self._detector: RTOnsetDetector | None = None

        # Hit counting logic
        self._hit_times: list[float] = []
        self._hit_window = 2.0  # seconds to consider hits for pattern detection
        self._min_hits_for_start = 4  # Changed from 3 to 4 as requested
        self._min_hits_for_stop = 2  # Require 2 hits to stop music
        self._music_playing = False
        self._music_start_time: float | None = None  # Track when music was started
        self._start_grace_period = 1.5  # seconds to ignore stop hits after starting music

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def start(self) -> None:
        """Start the audio worker."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Set status to active when worker starts
        if self.on_status_change is not None:
            self.on_status_change(True)

    def stop(self) -> None:
        """Stop the audio worker."""
        self._running = False
        # Set status to inactive when worker stops
        if self.on_status_change is not None:
            self.on_status_change(False)

    def sync_music_state(self, is_playing: bool) -> None:
        """Synchronize the internal music playing state with the actual player state."""
        logger.info("Synchronizing audio worker music state: %s", is_playing)
        self._music_playing = is_playing
        # Clear hit history when state changes to prevent false triggers
        self._hit_times.clear()
        # Reset grace period when manually stopped
        if not is_playing:
            self._music_start_time = None

    def _load_model(self) -> bool:
        """Load the trained model."""
        try:
            # Load model
            backbone = SincNetDSN(sr=self.sample_rate)
            pl_module = OnsetLightningModule.load_from_checkpoint(
                self.checkpoint_path, backbone=backbone, map_location=self.device
            )
            pl_module.eval()
            pl_module.to(self.device)
            model = pl_module.model
            model.eval()
            model.to(self.device)

            self._model = model
            self._detector = RTOnsetDetector(
                model=model,
                sample_rate=self.sample_rate,
                frame_size=self.frame_size,
                hop_size=self.hop_size,
                threshold=self.threshold,
                min_separation_s=self.min_separation_s,
                device=self.device,
            )
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    def _run(self) -> None:
        """Run the audio worker."""
        if not self._load_model():
            logger.error("Could not load model, audio worker stopping")
            self._running = False
            return

        # Audio input via sounddevice
        q: queue.Queue[np.ndarray] = queue.Queue()
        stop_flag = threading.Event()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning("[audio] %s", status)
            try:
                q.put_nowait(indata.copy())
            except queue.Full:
                pass

        stream_kwargs = dict(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.hop_size,
            callback=audio_callback,
        )
        if self.input_device is not None:
            try:
                stream_kwargs["device"] = int(self.input_device)
            except ValueError:
                stream_kwargs["device"] = self.input_device

        logger.info("Audio worker started - listening for drumstick hits")
        logger.info("Hit pattern: 4 hits = start music, 2 hits = stop music")

        try:
            with sd.InputStream(**stream_kwargs):
                while self._running and not stop_flag.is_set():
                    try:
                        data = q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if self._detector is None:
                        continue

                    onsets = self._detector.process_samples(data[:, 0] if data.ndim == 2 else data)

                    for t, p in onsets:
                        logger.info("Drumstick hit detected @ %.3fs (p=%.2f)", t, p)
                        self._process_hit(t)

        except Exception as e:
            logger.exception("Audio worker crashed: %s", e)
        finally:
            stop_flag.set()

    def _process_hit(self, hit_time: float) -> None:
        """Process a detected hit and determine music control actions."""
        current_time = time.time()

        # Add hit to recent hits list
        self._hit_times.append(current_time)

        # Remove old hits outside the window
        self._hit_times = [t for t in self._hit_times if current_time - t <= self._hit_window]

        hit_count = len(self._hit_times)
        logger.info("Recent hits in window: %d (music playing: %s)", hit_count, self._music_playing)

        # Determine action based on hit pattern
        if self._music_playing:
            # Check if we're in the grace period after starting music
            if (self._music_start_time is not None and
                current_time - self._music_start_time < self._start_grace_period):
                logger.info("Ignoring hit during grace period (%.1fs remaining)",
                           self._start_grace_period - (current_time - self._music_start_time))
                return

            # When music is playing, need 2 hits to stop it
            if hit_count >= self._min_hits_for_stop:
                logger.info("Stopping music (detected %d hits while playing)", hit_count)
                self._music_playing = False
                self._music_start_time = None
                self._hit_times.clear()  # Clear to prevent false start triggers
                self.on_stop_music()
                # Note: Don't change status here - worker is still active and listening
            else:
                logger.info("Hit detected while playing, need %d hits to stop (current: %d)",
                           self._min_hits_for_stop, hit_count)
        elif hit_count >= self._min_hits_for_start:
            # When music is not playing, need exactly 4 hits to start
            logger.info("Starting music (detected %d hits)", hit_count)
            self._music_playing = True
            self._music_start_time = current_time  # Record when music started
            self._hit_times.clear()  # Clear to prevent repeated start triggers
            self.on_start_music()
            # Note: Don't change status here - worker was already active


class RTOnsetDetector:
    """Real-time onset detector for drumstick hits."""

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
