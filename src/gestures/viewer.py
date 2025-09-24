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


def build_hand_entries(result: object) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    multi_hands = getattr(result, "multi_hand_landmarks", None)
    handedness = getattr(result, "multi_handedness", [])
    if not multi_hands:
        return entries
    for idx, landmarks in enumerate(multi_hands):
        label = "Unknown"
        score = 0.0
        if idx < len(handedness):
            try:
                hd = handedness[idx].classification[0]
                label = hd.label
                score = hd.score
            except Exception:
                logger.debug("Failed to read handedness classification", exc_info=True)
        norm_points = [(float(pt.x), float(pt.y)) for pt in landmarks.landmark]
        if not norm_points:
            continue
        cx_norm = sum(pt[0] for pt in norm_points) / float(len(norm_points))
        entries.append(
            {
                "label": label,
                "score": score,
                "lm": landmarks,
                "norm": norm_points,
                "cx": cx_norm,
            }
        )
    return entries


def select_hand_roles(
    entries: list[dict[str, object]],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not entries:
        return None, None

    def _pick_best(items: list[dict[str, object]]) -> dict[str, object] | None:
        if not items:
            return None
        return max(items, key=lambda e: float(e.get("score", 0.0)))

    rights = [e for e in entries if str(e.get("label", "")).lower().startswith("right")]
    lefts = [e for e in entries if str(e.get("label", "")).lower().startswith("left")]
    transport = _pick_best(rights)
    loop = _pick_best(lefts)

    ordered = sorted(entries, key=lambda e: float(e.get("cx", 0.0)))
    leftmost = ordered[0] if ordered else None
    rightmost = ordered[-1] if ordered else None

    if transport is None:
        transport = rightmost if rightmost is not None else leftmost

    if loop is None:
        candidates = [e for e in ordered if e is not transport]
        loop = candidates[0] if candidates else leftmost

    if loop is transport:
        loop = next((e for e in ordered if e is not transport), None)

    return transport, loop


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
                                     max_num_hands=2,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
    smoother = SimpleSmoother(window_size=3)

    prev_t = time.time()
    command_log: list[tuple[str, float]] = []
    last_toggle_time = 0.0
    last_gesture_toggle_time = 0.0
    seek_accum = 0.0
    loop_state = "OFF"  # OFF | A | AB
    loop_next = "Next: A"
    gesture_paused = True  # Track gesture control state (start paused)

    exit_key: int = 27 # ESC

    try:
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
                entries = build_hand_entries(res)
                primary, secondary = select_hand_roles(entries)

                # Primary (transport HUD)
                if primary is not None:
                    plm = primary["lm"]
                    landmarks = [(int(p.x * w), int(p.y * h)) for p in plm.landmark]
                    wrist = landmarks[0]
                    mcp = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
                    cx = float((wrist[0] + sum(j[0] for j in mcp)) / 5.0)
                    cy = float((wrist[1] + sum(j[1] for j in mcp)) / 5.0)

                    smoother.update(cx, cy, dt=dt)
                    vx, vy = smoother.get_velocity()

                    cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 0), 2)
                    cv2.putText(vis, f"vx={vx:.0f} vy={vy:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)
                    cv2.putText(vis, "ESC to close", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

                    # HUD gesture mimics
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

                    # V-sign detection for gesture control activation/deactivation (right hand)
                    try:
                        # Get normalized landmarks for gesture detection
                        norm_landmarks = [(float(p.x), float(p.y)) for p in plm.landmark]

                        # Hand size proxy: wrist (0) to middle MCP (9)
                        wrist_x, wrist_y = norm_landmarks[0]
                        mid_mcp_x, mid_mcp_y = norm_landmarks[9]
                        hand_size = max(
                            1e-3,
                            ((mid_mcp_x - wrist_x) ** 2 + (mid_mcp_y - wrist_y) ** 2) ** 0.5,
                        )

                        # Finger up checks
                        _, idx_tip_y = norm_landmarks[8]
                        _, idx_pip_y = norm_landmarks[6]
                        _, mid_tip_y = norm_landmarks[12]
                        _, mid_pip_y = norm_landmarks[10]
                        _, ring_tip_y = norm_landmarks[16]
                        _, ring_pip_y = norm_landmarks[14]
                        _, pky_tip_y = norm_landmarks[20]
                        _, pky_pip_y = norm_landmarks[18]

                        m = 0.02  # vertical margin
                        index_up = (idx_tip_y + m) < idx_pip_y
                        middle_up = (mid_tip_y + m) < mid_pip_y
                        ring_up = (ring_tip_y + m) < ring_pip_y
                        pinky_up = (pky_tip_y + m) < pky_pip_y

                        ix, iy = norm_landmarks[8]
                        mx, my = norm_landmarks[12]
                        separation = ((ix - mx) ** 2 + (iy - my) ** 2) ** 0.5
                        sep_ratio = separation / hand_size

                        is_v_sign = (
                            index_up and middle_up and (not ring_up) and (not pinky_up)
                            and (sep_ratio >= 0.25)  # separation threshold
                        )

                        # Toggle gesture pause/active state with right-hand V-sign
                        if is_v_sign and (now - last_gesture_toggle_time) > 0.8:
                            last_gesture_toggle_time = now
                            gesture_paused = not gesture_paused
                            status_text = "GESTURE PAUSED" if gesture_paused else "GESTURE ACTIVE"
                            command_log.append((status_text, now + 2.0))

                    except Exception:
                        logger.exception("Right-hand V-sign detection failed in viewer")

                # Secondary (left) V-sign -> loop HUD
                if secondary is not None:
                    slm = secondary["lm"]
                    lnorm = [(float(p.x), float(p.y)) for p in slm.landmark]
                    try:
                        # Hand size and finger-up checks
                        wx, wy = lnorm[0]
                        mx9, my9 = lnorm[9]
                        hand_size = max(1e-3, ((mx9 - wx) ** 2 + (my9 - wy) ** 2) ** 0.5)

                        _, idx_tip_y = lnorm[8]
                        _, idx_pip_y = lnorm[6]
                        _, mid_tip_y = lnorm[12]
                        _, mid_pip_y = lnorm[10]
                        _, ring_tip_y = lnorm[16]
                        _, ring_pip_y = lnorm[14]
                        _, pky_tip_y = lnorm[20]
                        _, pky_pip_y = lnorm[18]

                        m = 0.02
                        index_up = (idx_tip_y + m) < idx_pip_y
                        middle_up = (mid_tip_y + m) < mid_pip_y
                        ring_up = (ring_tip_y + m) < ring_pip_y
                        pinky_up = (pky_tip_y + m) < pky_pip_y

                        ix, iy = lnorm[8]
                        mx, my = lnorm[12]
                        separation = ((ix - mx) ** 2 + (iy - my) ** 2) ** 0.5
                        sep_ratio = separation / hand_size

                        is_v = index_up and middle_up and (not ring_up) and (not pinky_up) and (sep_ratio >= 0.25)
                        if is_v and (now - last_toggle_time) > 0.8:
                            last_toggle_time = now
                            next_state = {
                                "OFF": ("A", "LOOP: set A", "Next: B"),
                                "A": ("AB", "LOOP: set B", "Next: clear"),
                                "AB": ("OFF", "LOOP: clear", "Next: A"),
                            }
                            loop_state, action_text, loop_next = next_state.get(loop_state, ("A", "LOOP: set A", "Next: B"))
                            command_log.append((action_text, now + 1.5))
                    except Exception:
                        logger.exception("Left-hand V-sign detection failed in viewer")
            else:
                seek_accum *= 0.9

            # Draw recent recognized signals (top-right)
            now = time.time()
            command_log = [(txt, t) for (txt, t) in command_log if t > now]
            y0 = 30
            for idx, (txt, _) in enumerate(command_log[-6:]):
                cv2.putText(vis, txt, (w - 320, y0 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # Persistent loop state indicator (top-left under velocity)
            loop_text = {
                "OFF": "Loop: OFF",
                "A": "Loop: A set",
                "AB": "Loop: A-B active",
            }.get(loop_state, "Loop: OFF")
            cv2.putText(vis, loop_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(vis, loop_next, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 255), 1)

            # Persistent gesture control status indicator
            gesture_text = "Gestures: PAUSED" if gesture_paused else "Gestures: ACTIVE"
            gesture_color = (0, 0, 255) if gesture_paused else (0, 255, 0)  # Red if paused, green if active
            cv2.putText(vis, gesture_text, (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)

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


