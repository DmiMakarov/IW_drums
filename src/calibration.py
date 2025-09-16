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

from src.tracking.kalman import ConstantAccelerationKalman
from src.tracking.smoothing import SimpleSmoother

DEFAULT_SETTINGS = {
    "seek_sensitivity": 0.002,       # seconds per pixel
    "volume_sensitivity": 0.004,    # volume per pixel
    "raise_threshold": 0.25,        # relative height (0..1 from top)
    "min_velocity": 120.0,          # px/s threshold for gestures
}

logger = logging.getLogger(__name__)

def settings_path() -> str:
    """Get the path to the settings file."""
    root = Path.resolve(Path.cwd() / Path(os.pardir))
    return root / "iw_drums" / "calibration.json"


def load_settings() -> dict[str, Any]:
    """Load the settings from the file."""
    path = settings_path()

    if Path.exists(path):
        with Path.open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info("Loaded calibration settings from %s", path)
            return data

    return DEFAULT_SETTINGS.copy()


def save_settings(data: dict[str, Any]) -> None:
    """Save the settings to the file."""
    path = settings_path()

    logger.info("Saving calibration settings to %s", path)

    with Path.open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved calibration settings to %s", path)


def run_basic_calibration(camera: int | str = 0,  # noqa: C901, PLR0912, PLR0915
                          flip: bool = False,  # noqa: FBT001, FBT002
                          use_kalman: bool = False) -> dict[str, Any]:  # noqa: FBT001, FBT002
    """Interactive on-camera calibration.

    Procedure (rough, time-limited to keep UX simple):
    1) Detect hand and measure neutral height for 2 seconds -> set raise_threshold slightly above.
    2) Ask user to move hand up/down strongly for 3 seconds -> measure vy percentiles for volume_sensitivity.
    3) Ask user to move hand left/right strongly for 3 seconds -> measure vx percentiles for seek_sensitivity.
    4) Set min_velocity as moderate percentile of observed |v|.
    """
    cap = None
    break_key: int = 27

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
        logger.info("Calibration started (flip=%s, use_kalman=%s)", flip, use_kalman)

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

        def draw_ui(vis: np.ndarray, steps: list[str], current_idx: int, lines: list[str]) -> None:
            y = 24
            cv2.putText(vis, "Calibration", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y += 10
            # Steps list
            x0 = 10
            y0 = y + 10
            for i, step in enumerate(steps):
                color = (0, 255, 255) if i == current_idx else (180, 180, 180)
                cv2.putText(vis, step, (x0, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Extra right-side info
            x_info = 420
            y_info = 60
            for i, line in enumerate(lines):
                cv2.putText(vis, line, (x_info, y_info + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        steps = [
            "1) Raise hand 3s to start",
            "2) Neutral: keep still (2s)",
            "3) Move up/down (3s)",
            "4) Move left/right (3s)",
            "5) Done",
        ]

        # Phase 1: Wait for raised hand for 3 seconds
        current_step = 0
        hold_required = 3.0
        held_since: float | None = None
        last_t = time.time()
        while True:
            ok, frame, now_t, h = next_frame()
            if not ok:
                break
            dt = max(1e-3, now_t - last_t)
            last_t = now_t
            hand = get_hand(frame)
            rel_y = 1.0
            if hand is None:
                tracker_predict()
                held_since = None
            else:
                cx, cy = hand
                tracker_update(cx, cy, dt)
                rel_y = float(cy / max(1.0, float(h)))
                if rel_y < 0.25:  # provisional raise threshold for starting
                    if held_since is None:
                        held_since = now_t
                    elif (now_t - held_since) >= hold_required:
                        break
                else:
                    held_since = None

            # Draw UI
            vis = frame.copy()
            elapsed = 0.0 if held_since is None else (now_t - held_since)
            remaining = max(0.0, hold_required - elapsed)
            lines = [
                f"Hand: {'DETECTED' if hand is not None else 'NOT DETECTED'}",
                f"Raise to start: {remaining:.1f}s left",
                f"rel_y={rel_y:.2f} (<0.25 triggers)",
            ]
            draw_ui(vis, steps, current_step, lines)
            cv2.imshow("calibration", vis)
            if (cv2.waitKey(1) & 0xFF) == break_key:
                raise KeyboardInterrupt

        logger.info("Calibration start signal received")

        # Collect neutral height (2s)
        current_step = 1
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

            # UI
            vis = frame.copy()
            remaining = max(0.0, t_end - time.time())
            lines = [
                f"Collecting neutral... {remaining:.1f}s",
                f"Samples: {len(neutral_samples)}",
            ]
            draw_ui(vis, steps, current_step, lines)
            cv2.imshow("calibration", vis)
            if (cv2.waitKey(1) & 0xFF) == break_key:
                raise KeyboardInterrupt

        neutral_rel_y = float(np.median(neutral_samples)) if neutral_samples else 0.4
        raise_threshold = max(0.05, min(0.9, neutral_rel_y - 0.1))
        logger.info("Neutral rel_y=%.3f -> raise_threshold=%.3f", neutral_rel_y, raise_threshold)

        # Vertical motion collection for vy (3s)
        current_step = 2
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

            # UI
            vis = frame.copy()
            remaining = max(0.0, t_end - time.time())
            lines = [
                f"Move UP/DOWN... {remaining:.1f}s",
                f"|vy| p80 est updates: {len(vy_samples)}",
                f"vx={vx:.1f} vy={vy:.1f}",
            ]
            draw_ui(vis, steps, current_step, lines)
            cv2.imshow("calibration", vis)
            if (cv2.waitKey(1) & 0xFF) == break_key:
                raise KeyboardInterrupt

        # Horizontal motion collection for vx (3s)
        current_step = 3
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

            # UI
            vis = frame.copy()
            remaining = max(0.0, t_end - time.time())
            lines = [
                f"Move LEFT/RIGHT... {remaining:.1f}s",
                f"|vx| p80 est updates: {len(vx_samples)}",
                f"vx={vx:.1f} vy={vy:.1f}",
            ]
            draw_ui(vis, steps, current_step, lines)
            cv2.imshow("calibration", vis)
            if (cv2.waitKey(1) & 0xFF) == break_key:
                raise KeyboardInterrupt

        # Compute mappings
        def percentile(arr: list[float], p: float, default: float) -> float:
            if not arr:
                return default
            arr_sorted = np.sort(np.array(arr, dtype=np.float32))
            idx = int(max(0, min(len(arr_sorted) - 1, round((p / 100.0) * (len(arr_sorted) - 1)))))
            return float(arr_sorted[idx])

        p80_vy = percentile(vy_samples, 80.0, 300.0)
        p80_vx = percentile(vx_samples, 80.0, 300.0)
        # Target mapping update:
        # Map a strong full-motion (80th percentile velocity sustained for ~1s)
        # to a full-scale change of 0..100 for volume, and ~0..100 seconds for seek.
        # Note: downstream conversions multiply volume by 100 and apply dt, and seek is in seconds.
        # Therefore choose sensitivities so that: delta = v * sensitivity * dt * 100 (volume)
        # reaching ~100 when v = p80 over dt=1s → sensitivity_vol = 1 / p80.
        # For seek, match similar scale: 100s per 1s at p80 → sensitivity_seek = 100 / p80.
        volume_sensitivity = 20.0 / max(1.0, p80_vy)
        seek_sensitivity = 100.0 / max(1.0, p80_vx)
        # Min velocity so small jitters are ignored: set to 20th percentile of combined samples
        all_abs_v = vy_samples + vx_samples
        min_velocity = percentile(all_abs_v, 20.0, 120.0)
        logger.info("Derived: p80_vy=%.1f p80_vx=%.1f -> volume_sens=%.5f seek_sens=%.5f min_vel=%.1f",
                    p80_vy, p80_vx, volume_sensitivity, seek_sensitivity, min_velocity)

        data = {
            "seek_sensitivity": float(seek_sensitivity),
            "volume_sensitivity": float(volume_sensitivity),
            "raise_threshold": float(raise_threshold),
            "min_velocity": float(min_velocity),
        }
        # Show final results briefly
        current_step = 4
        ok, frame, _, _ = next_frame()
        if ok:
            vis = frame.copy()
            lines = [
                f"seek_sens={data['seek_sensitivity']:.5f}",
                f"vol_sens={data['volume_sensitivity']:.5f}",
                f"raise_th={data['raise_threshold']:.3f}",
                f"min_vel={data['min_velocity']:.1f}",
            ]
            draw_ui(vis, steps, current_step, lines)
            cv2.imshow("calibration", vis)
            cv2.waitKey(500)
        return data
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



if __name__ == "__main__":
    """Run interactive calibration as a CLI tool and save results."""
    try:
        data = run_basic_calibration()
        save_settings(data)
        print("Calibration completed and settings saved.")
    except KeyboardInterrupt:
        print("Calibration cancelled by user.")
    except Exception as e:  # noqa: BLE001
        print(f"Calibration failed: {e}")
