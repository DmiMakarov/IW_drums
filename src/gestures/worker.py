"""Background worker that reads camera and emits gesture callbacks."""
from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress
from typing import Callable

import cv2
import mediapipe as mp

from src.calibration import load_settings
from src.tracking.smoothing import SimpleSmoother

logger = logging.getLogger(__name__)


class GestureWorker:
    """Background worker that reads camera and emits gesture callbacks.

    The worker can optionally display a tracking visualization window.
    Heavy dependencies (OpenCV, MediaPipe) are imported lazily in the thread.
    """

    def __init__(
        self,
        on_toggle_play: Callable[[], None],
        on_seek_delta: Callable[[float], None],
        on_volume_delta: Callable[[float], None],
        camera: int | str = 0,
        show_window: bool = False, # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the gesture worker."""
        self.on_toggle_play = on_toggle_play
        self.on_seek_delta = on_seek_delta
        self.on_volume_delta = on_volume_delta
        self.camera = camera
        self.show_window = show_window

        self._thread: threading.Thread | None = None
        self._running = False

        # Threshold/constants
        self.SEEK_EMIT_THRESHOLD = 0.05  # seconds
        self._last_toggle_time = 0.0
        self._toggle_cooldown = 0.35  # seconds

        # Settings (support both src.* and package-relative imports)
        s = load_settings()
        self.seek_sensitivity = float(s.get("seek_sensitivity", 0.002))
        self.volume_sensitivity = float(s.get("volume_sensitivity", 0.004))
        self.raise_threshold = float(s.get("raise_threshold", 0.25))
        self.min_velocity = float(s.get("min_velocity", 120.0))

    def start(self) -> None:
        """Start the gesture worker."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the gesture worker."""
        self._running = False

    def _run(self) -> None:  # noqa: C901, PLR0915
        """Run the gesture worker."""
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            logging.getLogger(__name__).warning("Could not open camera %s for gestures", self.camera)
            self._running = False
            return

        hands = mp.solutions.hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

        smoother = SimpleSmoother(window_size=3)
        prev_t = time.time()
        seek_accum = 0.0

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(frame_rgb)

                now = time.time()
                dt = max(1e-3, now - prev_t)
                prev_t = now

                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    landmarks = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                    wrist = landmarks[0]
                    mcp = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
                    cx = float((wrist[0] + sum(j[0] for j in mcp)) / 5.0)
                    cy = float((wrist[1] + sum(j[1] for j in mcp)) / 5.0)

                    smoother.update(cx, cy, dt=dt)
                    vx, vy = smoother.get_velocity()  # px/s

                    # Toggle play on raise
                    rel_y = cy / max(1.0, float(h))
                    if rel_y < self.raise_threshold and (now - self._last_toggle_time) > self._toggle_cooldown:
                        self._last_toggle_time = now
                        logger.info("Raise threshold: %s", rel_y)
                        self.on_toggle_play()

                    # Seek by horizontal velocity
                    if abs(vx) > self.min_velocity:
                        seek_delta = vx * self.seek_sensitivity * dt  # seconds
                        seek_accum += seek_delta
                        if abs(seek_accum) >= self.SEEK_EMIT_THRESHOLD:
                            logger.info("Seek delta: %s", seek_accum)
                            self.on_seek_delta(seek_accum)
                            seek_accum = 0.0

                    # Volume by vertical velocity (invert so up increases)
                    if abs(vy) > self.min_velocity:
                        logger.info("Vertical velocity: %s", vy)
                        vol_delta = (-vy) * self.volume_sensitivity * dt * 100.0  # map to 0..100 scale
                        if abs(vol_delta) >= 1.0:
                            logger.info("Volume delta: %s", vol_delta)
                            self.on_volume_delta(vol_delta)
                else:
                    seek_accum *= 0.9

                # Do not call cv2.imshow/cv2.waitKey in worker thread

        except Exception:
            logging.getLogger(__name__).exception("Gesture worker crashed")
        finally:
            with suppress(Exception):
                hands.close()
            cap.release()
            # No GUI cleanup needed here


