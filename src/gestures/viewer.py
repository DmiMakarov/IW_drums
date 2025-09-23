"""Standalone gesture tracking viewer.

Runs in its own process to display a camera window with hand tracking overlay.
This avoids OpenCV GUI threading issues inside the main Tkinter app.
"""
from __future__ import annotations

import argparse
import logging
import time

import cv2
import mediapipe as mp

from src.calibration import load_settings
from src.tracking.smoothing import SimpleSmoother

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the arguments."""
    ap = argparse.ArgumentParser(description="Gesture tracking viewer")
    ap.add_argument("--camera", default=0, help="Camera index or source path")
    ap.add_argument("--flip", action="store_true", help="Flip horizontally for selfie view")
    return ap.parse_args()


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    """Run the program."""
    args = parse_args()
    settings = load_settings()
    seek_sensitivity = float(settings.get("seek_sensitivity", 0.002))
    volume_sensitivity = float(settings.get("volume_sensitivity", 0.004))
    raise_threshold = float(settings.get("raise_threshold", 0.25))
    min_velocity = float(settings.get("min_velocity", 120.0))
    seek_emit_threshold = 0.05
    toggle_cooldown = 0.35
    try:
        cam_idx = int(args.camera)
        cap = cv2.VideoCapture(cam_idx)
    except ValueError:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        logger.error("Could not open camera: %s", args.camera)
        return

    hands = mp.solutions.hands.Hands(static_image_mode=False,
                                     max_num_hands=1,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
    smoother = SimpleSmoother(window_size=3)

    prev_t = time.time()
    command_log: list[tuple[str, float]] = []
    last_toggle_time = 0.0
    seek_accum = 0.0

    exit_key: int = 27 # ESC

    try:
        # Ensure window is created explicitly before first imshow
        try:
            cv2.namedWindow("Gesture Viewer", cv2.WINDOW_NORMAL)
        except Exception:
            logger.exception("Could not create named window; continuing with imshow")
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)

            now = time.time()
            dt = max(1e-3, now - prev_t)
            prev_t = now

            vis = frame

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                landmarks = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                wrist = landmarks[0]
                mcp = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
                cx = float((wrist[0] + sum(j[0] for j in mcp)) / 5.0)
                cy = float((wrist[1] + sum(j[1] for j in mcp)) / 5.0)

                smoother.update(cx, cy, dt=dt)
                vx, vy = smoother.get_velocity()

                cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 0), 2)
                cv2.putText(vis, f"vx={vx:.0f} vy={vy:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)
                cv2.putText(vis, "ESC to close", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

                # Recognize gestures the same way as worker and show HUD
                rel_y = cy / max(1.0, float(h))
                if rel_y < raise_threshold and (now - last_toggle_time) > toggle_cooldown:
                    last_toggle_time = now
                    command_log.append(("TOGGLE PLAY/PAUSE", now + 1.5))

                if abs(vx) > min_velocity:
                    seek_delta = vx * seek_sensitivity * dt
                    seek_accum += seek_delta
                    if abs(seek_accum) >= seek_emit_threshold:
                        sign = "+" if seek_accum >= 0 else ""
                        command_log.append((f"SEEK {sign}{seek_accum:.2f}s", now + 1.2))
                        seek_accum = 0.0

                if abs(vy) > min_velocity:
                    vol_delta = (-vy) * volume_sensitivity * dt * 100.0
                    if abs(vol_delta) >= 1.0:
                        sign = "+" if vol_delta >= 0 else ""
                        command_log.append((f"VOLUME {sign}{int(vol_delta)}", now + 0.8))
            else:
                seek_accum *= 0.9

            # Draw recent recognized signals (top-right)
            now = time.time()
            command_log = [(txt, t) for (txt, t) in command_log if t > now]
            y0 = 30
            for idx, (txt, _) in enumerate(command_log[-6:]):
                cv2.putText(vis, txt, (w - 320, y0 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("Gesture Viewer", vis)
            if cv2.waitKey(1) & 0xFF == exit_key:
                break
    except Exception:
        logger.exception("Viewer crashed")
    finally:
        try:
            hands.close()
        except Exception:
            logger.exception("Could not close hands")
        cap.release()
        try:
            cv2.destroyWindow("Gesture Viewer")
        except Exception:
            logger.exception("Could not destroy window")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()


