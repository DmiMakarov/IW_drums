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
        on_loop_set_a: Callable[[], None] | None = None,
        on_loop_set_b: Callable[[], None] | None = None,
        on_loop_clear: Callable[[], None] | None = None,
        on_status_change: Callable[[bool], None] | None = None,
        camera: int | str = 0,
        show_window: bool = False, # noqa: FBT001, FBT002
        initial_paused: bool = True,
    ) -> None:
        """Initialize the gesture worker."""
        self.on_toggle_play = on_toggle_play
        self.on_seek_delta = on_seek_delta
        self.on_volume_delta = on_volume_delta
        self.on_loop_set_a = on_loop_set_a
        self.on_loop_set_b = on_loop_set_b
        self.on_loop_clear = on_loop_clear
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
        self._paused = bool(initial_paused)
        self._last_gesture_toggle_time = 0.0
        self._toggle_cooldown_gesture = 0.8  # seconds
        self._last_loop_time = 0.0
        self._loop_cycle = 0  # A -> B -> clear cycling for left-hand V-sign


        # Settings (support both src.* and package-relative imports)
        s = load_settings()
        self.seek_sensitivity = float(s.get("seek_sensitivity", 0.002))
        self.volume_sensitivity = float(s.get("volume_sensitivity", 0.004))
        self.raise_threshold = float(s.get("raise_threshold", 0.25))
        self.min_velocity = float(s.get("min_velocity", 120.0))
        self.folded_count_threshold = int(s.get("folded_count_threshold", 3))
        # Hand role configuration: set to true if your camera view is mirrored
        self.swap_hands = bool(s.get("swap_hands", False))
        # Fist detection thresholds (normalized by hand size)
        # finger_fold_threshold: tip-to-MCP distance / hand_size for a finger to be considered folded
        # spread_threshold: index-MCP to pinky-MCP distance / hand_size to ensure palm is not widely spread
        self.finger_fold_threshold = float(s.get("finger_fold_threshold", 0.35))
        self.spread_threshold = float(s.get("spread_threshold", 1.4))
        # V-sign detection tuning
        # Required separation (tips) between index and middle relative to hand size
        self.v_separation_threshold = float(s.get("v_separation_threshold", 0.25))
        # Vertical margin so a finger counts as "up" if tip is this much above PIP
        self.v_vertical_margin = float(s.get("v_vertical_margin", 0.02))

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

    def _prepare_hand_entries(self, result: object) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        multi_hands = getattr(result, "multi_hand_landmarks", None)
        handedness = getattr(result, "multi_handedness", [])
        if not multi_hands:
            return entries
        for idx, landmarks in enumerate(multi_hands):
            label = "Unknown"
            score = 0.0
            if idx < len(handedness):
                with suppress(Exception):
                    label = handedness[idx].classification[0].label
                    score = handedness[idx].classification[0].score
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

    def _select_hand_roles(
        self,
        entries: list[dict[str, object]],
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        if not entries:
            return None, None
        rights = [e for e in entries if str(e.get("label", "")).lower().startswith("right")]
        lefts = [e for e in entries if str(e.get("label", "")).lower().startswith("left")]

        def _pick_best(items: list[dict[str, object]]) -> dict[str, object] | None:
            if not items:
                return None
            return max(items, key=lambda e: float(e.get("score", 0.0)))

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

        if self.swap_hands:
            transport, loop = loop, transport

        if loop is transport:
            loop = next((e for e in ordered if e is not transport), None)

        return transport, loop

    def _run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Run the gesture worker."""
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            logging.getLogger(__name__).warning("Could not open camera %s for gestures", self.camera)
            self._running = False
            return

        hands = mp.solutions.hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
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

                has_landmarks = bool(res.multi_hand_landmarks)
                if has_landmarks:
                    entries = self._prepare_hand_entries(res)
                    transport_entry, loop_entry = self._select_hand_roles(entries)

                    if self.swap_hands:
                        transport_entry, loop_entry = loop_entry, transport_entry
                        if entries and self.on_status_change is not None:
                            is_active = not self._paused and transport_entry is not None
                            try:
                                self.on_status_change(is_active)
                            except Exception:
                                logger.exception("on_status_change callback failed")
                    else:
                        if entries and self.on_status_change is not None:
                            is_active = not self._paused and transport_entry is not None
                            try:
                                self.on_status_change(is_active)
                            except Exception:
                                logger.exception("on_status_change callback failed")

                    # Transport controls hand
                    if transport_entry is not None:
                        plm = transport_entry["lm"]
                        assert plm is not None
                        landmarks_px = [(int(p.x * w), int(p.y * h)) for p in plm.landmark]
                        wrist_px = landmarks_px[0]
                        mcp_px = [landmarks_px[5], landmarks_px[9], landmarks_px[13], landmarks_px[17]]
                        cx = float((wrist_px[0] + sum(j[0] for j in mcp_px)) / 5.0)
                        cy = float((wrist_px[1] + sum(j[1] for j in mcp_px)) / 5.0)

                        smoother.update(cx, cy, dt=dt)
                        vx, vy = smoother.get_velocity()  # px/s

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

                        # V-sign detection for activating/deactivating gesture control
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

                            m = self.v_vertical_margin
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
                                and (sep_ratio >= self.v_separation_threshold)
                            )

                            # Toggle gesture pause/active state with right-hand V-sign
                            if is_v_sign and (now - self._last_gesture_toggle_time) > self._toggle_cooldown_gesture:
                                self._last_gesture_toggle_time = now
                                self._paused = not self._paused
                                status = "Paused" if self._paused else "Active"
                                logger.info("Gesture control toggled: %s (right-hand V)", status)

                                # Notify status change
                                if self.on_status_change is not None:
                                    try:
                                        self.on_status_change(not self._paused)
                                    except Exception:
                                        logger.exception("on_status_change callback failed")

                        except Exception:
                            logger.exception("Transport-hand V-sign detection failed")

                    # Loop control hand (opposite side)
                    if loop_entry is not None:
                        slm = loop_entry["lm"]
                        norm_landmarks = loop_entry["norm"]
                        assert isinstance(norm_landmarks, list)
                        # Keep normalized landmarks for gesture logic
                        landmarks = norm_landmarks

                        try:
                            # Hand size proxy: wrist (0) to middle MCP (9)
                            wrist_x, wrist_y = landmarks[0]
                            mid_mcp_x, mid_mcp_y = landmarks[9]
                            hand_size = max(
                                1e-3,
                                ((mid_mcp_x - wrist_x) ** 2 + (mid_mcp_y - wrist_y) ** 2) ** 0.5,
                            )

                            # Finger up checks
                            _, idx_tip_y = landmarks[8]
                            _, idx_pip_y = landmarks[6]
                            _, mid_tip_y = landmarks[12]
                            _, mid_pip_y = landmarks[10]
                            _, ring_tip_y = landmarks[16]
                            _, ring_pip_y = landmarks[14]
                            _, pky_tip_y = landmarks[20]
                            _, pky_pip_y = landmarks[18]

                            m = self.v_vertical_margin
                            index_up = (idx_tip_y + m) < idx_pip_y
                            middle_up = (mid_tip_y + m) < mid_pip_y
                            ring_up = (ring_tip_y + m) < ring_pip_y
                            pinky_up = (pky_tip_y + m) < pky_pip_y

                            ix, iy = landmarks[8]
                            mx, my = landmarks[12]
                            separation = ((ix - mx) ** 2 + (iy - my) ** 2) ** 0.5
                            sep_ratio = separation / hand_size

                            is_v_sign = (
                                index_up and middle_up and (not ring_up) and (not pinky_up)
                                and (sep_ratio >= self.v_separation_threshold)
                            )

                            if is_v_sign and (now - self._last_loop_time) > self._toggle_cooldown_gesture:
                                self._last_loop_time = now
                                # Left-hand V-sign cycles loop points: A -> B -> clear
                                if self.on_loop_set_a and self.on_loop_set_b:
                                    cnt = self._loop_cycle
                                    action = cnt % 3
                                    if action == 0:
                                        logger.info("Loop: set point A (left-hand V)")
                                        try:
                                            self.on_loop_set_a()
                                        except Exception:
                                            logger.exception("on_loop_set_a callback failed")
                                    elif action == 1:
                                        logger.info("Loop: set point B (left-hand V)")
                                        try:
                                            self.on_loop_set_b()
                                        except Exception:
                                            logger.exception("on_loop_set_b callback failed")
                                    else:
                                        if self.on_loop_clear is not None:
                                            logger.info("Loop: clear (left-hand V)")
                                            try:
                                                self.on_loop_clear()
                                            except Exception:
                                                logger.exception("on_loop_clear callback failed")
                                    self._loop_cycle = cnt + 1
                        except Exception:
                            logger.exception("Secondary-hand V-sign detection failed")
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


