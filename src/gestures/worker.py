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

    def __init__(  # noqa: PLR0913
        self,
        on_toggle_play: Callable[[], None],
        on_seek_delta: Callable[[float], None],
        on_volume_delta: Callable[[float], None],
        on_status_change: Callable[[bool], None] | None = None,
        camera: int | str = 0,
        show_window: bool = False, # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the gesture worker."""
        self.on_toggle_play = on_toggle_play
        self.on_seek_delta = on_seek_delta
        self.on_volume_delta = on_volume_delta
        self.camera = camera
        self.show_window = show_window
        self.on_status_change = on_status_change

        self._thread: threading.Thread | None = None
        self._running = False

        # Threshold/constants
        self.SEEK_EMIT_THRESHOLD = 0.05  # seconds
        self._last_toggle_time = 0.0
        self._toggle_cooldown = 0.35  # seconds

        # Paused mode toggled by "V" gesture (index+middle extended)
        self._paused = False
        self._last_toggle_time = 0.0
        self._toggle_cooldown_gesture = 0.8  # seconds

        # Settings (support both src.* and package-relative imports)
        s = load_settings()
        self.seek_sensitivity = float(s.get("seek_sensitivity", 0.002))
        self.volume_sensitivity = float(s.get("volume_sensitivity", 0.004))
        self.raise_threshold = float(s.get("raise_threshold", 0.25))
        self.min_velocity = float(s.get("min_velocity", 120.0))
        self.folded_count_threshold = int(s.get("folded_count_threshold", 3))
        # Fist detection thresholds (normalized by hand size)
        # finger_fold_threshold: tip-to-MCP distance / hand_size for a finger to be considered folded
        # spread_threshold: index-MCP to pinky-MCP distance / hand_size to ensure palm is not widely spread
        self.finger_fold_threshold = float(s.get("finger_fold_threshold", 0.35))
        self.spread_threshold = float(s.get("spread_threshold", 1.4))
        # V-sign detection: required separation (tips) between index and middle relative to hand size
        self.v_separation_threshold = float(s.get("v_separation_threshold", 0.5))

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

    def _run(self) -> None:  # noqa: C901, PLR0912, PLR0915
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
                    # Keep both pixel coords and normalized floats for distances
                    landmarks_px = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                    landmarks = [(float(p.x), float(p.y)) for p in lm.landmark]
                    wrist_px = landmarks_px[0]
                    mcp_px = [landmarks_px[5], landmarks_px[9], landmarks_px[13], landmarks_px[17]]
                    cx = float((wrist_px[0] + sum(j[0] for j in mcp_px)) / 5.0)
                    cy = float((wrist_px[1] + sum(j[1] for j in mcp_px)) / 5.0)

                    smoother.update(cx, cy, dt=dt)
                    vx, vy = smoother.get_velocity()  # px/s

                    # V-sign detection to toggle paused mode
                    try:
                        # Hand size proxy: wrist (0) to middle MCP (9)
                        wrist = landmarks[0]
                        mid_mcp = landmarks[9]
                        hand_size = max(1e-3, ((mid_mcp[0] - wrist[0]) ** 2 + (mid_mcp[1] - wrist[1]) ** 2) ** 0.5)

                        # Per-finger tip-to-MCP distances (thumb uses 4-2, others tip-[MCP])
                        # Compute folded flags per finger without inner defs (for linter)
                        ix_tip_x, ix_tip_y = landmarks[8]
                        ix_mcp_x, ix_mcp_y = landmarks[5]
                        mid_tip_x, mid_tip_y = landmarks[12]
                        mid_mcp_x, mid_mcp_y = landmarks[9]
                        ring_tip_x, ring_tip_y = landmarks[16]
                        ring_mcp_x, ring_mcp_y = landmarks[13]
                        pky_tip_x, pky_tip_y = landmarks[20]
                        pky_mcp_x, pky_mcp_y = landmarks[17]

                        ix_ratio = (
                            ((ix_tip_x - ix_mcp_x) ** 2 + (ix_tip_y - ix_mcp_y) ** 2) ** 0.5
                        ) / hand_size
                        mid_ratio = (
                            ((mid_tip_x - mid_mcp_x) ** 2 + (mid_tip_y - mid_mcp_y) ** 2) ** 0.5
                        ) / hand_size
                        ring_ratio = (
                            ((ring_tip_x - ring_mcp_x) ** 2 + (ring_tip_y - ring_mcp_y) ** 2) ** 0.5
                        ) / hand_size
                        pky_ratio = (
                            ((pky_tip_x - pky_mcp_x) ** 2 + (pky_tip_y - pky_mcp_y) ** 2) ** 0.5
                        ) / hand_size

                        index_folded = ix_ratio < self.finger_fold_threshold
                        middle_folded = mid_ratio < self.finger_fold_threshold
                        ring_folded = ring_ratio < self.finger_fold_threshold
                        pinky_folded = pky_ratio < self.finger_fold_threshold

                        # Separation between index and middle tips
                        ix, iy = landmarks[8]
                        mx, my = landmarks[12]
                        separation = ((ix - mx) ** 2 + (iy - my) ** 2) ** 0.5
                        sep_ratio = separation / hand_size

                        # V-sign criteria: index & middle extended, ring & pinky folded, thumb don't-care
                        is_v_sign = (
                            (not index_folded)
                            and (not middle_folded)
                            and ring_folded
                            and pinky_folded
                            and (sep_ratio >= self.v_separation_threshold)
                        )

                        if is_v_sign and (now - self._last_toggle_time) > self._toggle_cooldown_gesture:
                            self._last_toggle_time = now
                            self._paused = not self._paused
                            logger.info(
                                "V-sign detected (sep=%.2f>=%.2f, ring/pinky folded).",
                                sep_ratio,
                                self.v_separation_threshold,
                            )
                            logger.info("Gestures %s", "PAUSED" if self._paused else "ACTIVE")
                            if self.on_status_change is not None:
                                try:
                                    self.on_status_change(not self._paused)
                                except Exception:
                                    logger.exception("on_status_change callback failed")
                    except Exception:
                        # If any keypoint missing, ignore silently
                        logger.exception("V-sign gesture detection failed")

                    # Toggle play on raise (only when not paused)
                    rel_y = cy / max(1.0, float(h))
                    cond: bool = (not self._paused) and \
                                 rel_y < self.raise_threshold and \
                                 (now - self._last_toggle_time) > self._toggle_cooldown
                    if cond:
                        self._last_toggle_time = now
                        logger.info("Raise threshold: %s", rel_y)
                        self.on_toggle_play()

                    # Seek by horizontal velocity (only when not paused)
                    if (not self._paused) and abs(vx) > self.min_velocity:
                        seek_delta = vx * self.seek_sensitivity * dt  # seconds
                        seek_accum += seek_delta
                        if abs(seek_accum) >= self.SEEK_EMIT_THRESHOLD:
                            logger.info("Seek delta: %s", -seek_accum)
                            self.on_seek_delta(-seek_accum)
                            seek_accum = 0.0

                    # Volume by vertical velocity (invert so up increases), only when not paused
                    if (not self._paused) and abs(vy) > self.min_velocity:
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


