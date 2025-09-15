"""Calibration utilities for gesture-to-control mapping.

This module measures per-user motion ranges and computes mappings:
- Vertical velocity to volume delta
- Horizontal velocity to seek/rewind delta
- Raise threshold for play/pause

Settings are persisted as a simple JSON next to the project root.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from tracking.kalman import ConstantAccelerationKalman
from tracking.smoothing import SimpleSmoother

DEFAULT_SETTINGS = {
    "seek_sensitivity": 0.002,       # seconds per pixel
    "volume_sensitivity": 0.004,    # volume per pixel
    "raise_threshold": 0.25,        # relative height (0..1 from top)
    "min_velocity": 120.0,          # px/s threshold for gestures
}

logger = logging.getLogger(__name__)

def settings_path() -> str:
    """Get the path to the settings file."""
    root = Path.resolve(Path.parent / Path(os.pardir))
    return root /  "calibration.json"


def load_settings() -> dict[str, Any]:
    """Load the settings from the file."""
    path = settings_path()

    if Path.exists(path):
        with Path.open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return DEFAULT_SETTINGS.copy()


def save_settings(data: dict[str, Any]) -> None:
    """Save the settings to the file."""
    path = settings_path()

    with Path.open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_basic_calibration(camera: int | str = 0,  # noqa: C901, PLR0912, PLR0915
                          flip: bool = False,  # noqa: FBT001, FBT002
                          use_kalman: bool = True) -> dict[str, Any]:  # noqa: FBT001, FBT002
    """Interactive on-camera calibration.

    Procedure (rough, time-limited to keep UX simple):
    1) Detect hand and measure neutral height for 2 seconds -> set raise_threshold slightly above.
    2) Ask user to move hand up/down strongly for 3 seconds -> measure vy percentiles for volume_sensitivity.
    3) Ask user to move hand left/right strongly for 3 seconds -> measure vx percentiles for seek_sensitivity.
    4) Set min_velocity as moderate percentile of observed |v|.
    """
    cap = None

    try:
        try:
            cam_idx = int(camera)
            cap = cv2.VideoCapture(cam_idx)
        except ValueError:
            cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            msg: str = "Could not open camera for calibration"
            raise RuntimeError(msg)
        cap.set(cv2.CAP_PROP_FPS, 120)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Choose tracker per preference
        if use_kalman:
            kf = ConstantAccelerationKalman(dt=1/120.0, process_var=25.0, meas_var=10.0)

            def tracker_update(cx: float, cy: float, dt: float) -> None:  # noqa: ARG001
                kf.update(np.array([cx, cy], dtype=np.float32), r_scale=1.0)

            def tracker_predict() -> None:
                kf.predict()

            def tracker_velocity() -> tuple[float, float]:
                return kf.get_velocity()
        else:
            sm = SimpleSmoother(window_size=3)

            def tracker_update(cx: float, cy: float, dt: float) -> None:
                sm.update(cx, cy, dt=dt)

            def tracker_predict() -> None:
                # No prediction step for the simple smoother
                return None

            def tracker_velocity() -> tuple[float, float]:
                return sm.get_velocity()

        def get_hand(frame_bgr: np.ndarray) -> tuple[float, float] | None:
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)
            if not res.multi_hand_landmarks:
                return None
            lm = res.multi_hand_landmarks[0]
            landmarks = []
            for landmark in lm.landmark:
                x = landmark.x * w
                y = landmark.y * h
                landmarks.append((x, y))
            wrist = landmarks[0]
            mcp = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
            cx = (wrist[0] + sum(j[0] for j in mcp)) / 5
            cy = (wrist[1] + sum(j[1] for j in mcp)) / 5
            return cx, cy

        def next_frame() -> tuple[bool, np.ndarray, float, int]:
            ok, frame = cap.read()
            if not ok:
                return False, frame, 0.0, 0
            if flip:
                frame = cv2.flip(frame, 1)
            return True, frame, time.time(), frame.shape[0]

        # Collect neutral height
        neutral_samples = []
        t_end = time.time() + 2.0
        last_t = time.time()
        while time.time() < t_end:
            ok, frame, now_t, h = next_frame()
            if not ok:
                break
            dt = max(1e-3, now_t - last_t)
            last_t = now_t
            hand = get_hand(frame)
            if hand is None:
                tracker_predict()
            else:
                cx, cy = hand
                tracker_update(cx, cy, dt)
                neutral_samples.append(float(cy / max(1.0, float(h))))

        neutral_rel_y = float(np.median(neutral_samples)) if neutral_samples else 0.4
        raise_threshold = max(0.05, min(0.9, neutral_rel_y - 0.1))

        # Vertical motion collection for vy
        vy_samples = []
        t_end = time.time() + 3.0
        last_t = time.time()
        while time.time() < t_end:
            ok, frame, now_t, _ = next_frame()
            if not ok:
                break
            dt = max(1e-3, now_t - last_t)
            last_t = now_t
            hand = get_hand(frame)
            if hand is None:
                tracker_predict()
            else:
                cx, cy = hand
                tracker_update(cx, cy, dt)
            vx, vy = tracker_velocity()
            vy_samples.append(abs(float(vy)))

        # Horizontal motion collection for vx
        vx_samples = []
        t_end = time.time() + 3.0
        last_t = time.time()
        while time.time() < t_end:
            ok, frame, now_t, _ = next_frame()
            if not ok:
                break
            dt = max(1e-3, now_t - last_t)
            last_t = now_t
            hand = get_hand(frame)
            if hand is None:
                tracker_predict()
            else:
                cx, cy = hand
                tracker_update(cx, cy, dt)
            vx, vy = tracker_velocity()
            vx_samples.append(abs(float(vx)))

        # Compute mappings
        def percentile(arr: list[float], p: float, default: float) -> float:
            if not arr:
                return default
            arr_sorted = np.sort(np.array(arr, dtype=np.float32))
            idx = int(max(0, min(len(arr_sorted) - 1, round((p / 100.0) * (len(arr_sorted) - 1)))))
            return float(arr_sorted[idx])

        p80_vy = percentile(vy_samples, 80.0, 300.0)
        p80_vx = percentile(vx_samples, 80.0, 300.0)
        # Target mapping: 80th percentile motion over 1 second yields ~0.5 volume change and ~1.0s seek
        volume_sensitivity = 0.5 / max(1.0, p80_vy)
        seek_sensitivity = 1.0 / max(1.0, p80_vx)
        # Min velocity so small jitters are ignored: set to 20th percentile of combined samples
        all_abs_v = vy_samples + vx_samples
        min_velocity = percentile(all_abs_v, 20.0, 120.0)

        return {
            "seek_sensitivity": float(seek_sensitivity),
            "volume_sensitivity": float(volume_sensitivity),
            "raise_threshold": float(raise_threshold),
            "min_velocity": float(min_velocity),
        }
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not release camera: %s", e)
        try:
            cv2.destroyAllWindows()
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not destroy windows: %s", e)


def ensure_calibrated(allow_interactive: bool, # noqa: FBT001
                     camera: int | str = 0,
                     flip: bool = False, # noqa: FBT001, FBT002
                     use_kalman: bool = True) -> dict[str, Any]:  # noqa: FBT001, FBT002
    """Load existing settings; if missing or interactive requested, run calibration and save.

    use_kalman controls whether calibration collects velocities via Kalman or simple smoothing.
    """
    existing = load_settings()
    if allow_interactive or existing == DEFAULT_SETTINGS:
        data = run_basic_calibration(camera=camera, flip=flip, use_kalman=use_kalman)
        save_settings(data)
        return data
    return existing


