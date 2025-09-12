"""Prototype for a hand gesture controller.

It uses MediaPipe Hands for hand tracking and Kalman filter for smoothing.
It also uses a hit detector to detect when a hand is hit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

from tracking.kalman import ConstantAccelerationKalman


class SimpleSmoother:
    """Simple smoother for hand tracking."""

    def __init__(self, window_size: int=3) -> None:
        """Initialize the smoother."""
        self.positions = []
        self.velocities = []
        self.window_size = window_size
        self.last_time = None

    def update(self, x: float, y: float, dt: float=1/120.0) -> tuple[float, float]:
        """Update the smoother."""
        dimension: int = 2
        self.positions.append((x, y))
        if len(self.positions) > self.window_size:
            self.positions.pop(0)

        # Calculate velocity from recent positions
        if len(self.positions) >= dimension:
            # velocity in pixels/second using average over window
            total_dt = max(1e-6, (len(self.positions) - 1) * dt)
            vx = (self.positions[-1][0] - self.positions[0][0]) / total_dt
            vy = (self.positions[-1][1] - self.positions[0][1]) / total_dt
            self.velocities.append((vx, vy))

            if len(self.velocities) > dimension:
                self.velocities.pop(0)

        # Return smoothed position
        avg_x = sum(p[0] for p in self.positions) / len(self.positions)
        avg_y = sum(p[1] for p in self.positions) / len(self.positions)

        return avg_x, avg_y

    def get_position(self) -> tuple[float, float]:
        """Get the smoothed position."""
        if not self.positions:
            return 0, 0
        avg_x = sum(p[0] for p in self.positions) / len(self.positions)
        avg_y = sum(p[1] for p in self.positions) / len(self.positions)
        return avg_x, avg_y

    def get_velocity(self) -> tuple[float, float]:
        """Get the smoothed velocity."""
        if not self.velocities:
            return 0, 0
        avg_vx = sum(v[0] for v in self.velocities) / len(self.velocities)
        avg_vy = sum(v[1] for v in self.velocities) / len(self.velocities)
        return avg_vx, avg_vy


def parse_args() -> argparse.Namespace:
    """Parse the arguments."""
    ap = argparse.ArgumentParser(description="Hand gesture controller (MediaPipe Hands + Kalman)")
    ap.add_argument("--source", default=0, help="Camera index or video path")
    ap.add_argument("--display", action="store_true", help="Show visualization window")
    ap.add_argument("--seek_sensitivity", type=float, default=0.002, help="Seconds per pixel of horizontal move")
    ap.add_argument("--volume_sensitivity", type=float, default=0.004, help="Volume per pixel of vertical move")
    ap.add_argument("--raise_threshold", type=float, default=0.25,
                    help="Relative height to consider hand raised (0..1 from top)")
    ap.add_argument("--min_velocity", type=float, default=120.0,
                    help="Minimum velocity (pixels/second) to trigger gestures")
    ap.add_argument("--process_var", type=float, default=25.0,
                    help="Kalman process variance (higher = more responsive)")
    ap.add_argument("--meas_var", type=float, default=10.0,
                    help="Kalman measurement variance (lower = trust MediaPipe more)")
    ap.add_argument("--no_kalman", action="store_true",
                    help="Use simple smoothing instead of Kalman filter")
    ap.add_argument("--smooth_frames", type=int, default=3,
                    help="Number of frames for simple smoothing")
    ap.add_argument("--flip", action="store_true",
                    help="Mirror camera for selfie view")

    return ap.parse_args()


def open_source(src: str) -> cv2.VideoCapture:
    """Open the source."""
    try:
        src_int = int(src)
        cap = cv2.VideoCapture(src_int)
    except ValueError:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        msg: str = f"Could not open source: {src}"
        raise RuntimeError(msg)
    # Try to set higher FPS if webcam
    cap.set(cv2.CAP_PROP_FPS, 120)
    return cap

def get_hand_landmarks(hands: mp.solutions.hands.Hands, frame_bgr: np.ndarray) -> tuple[float, float, list] | None:
    """Return pixel coordinates (cx, cy) of hand center and full landmark list.

    Uses MediaPipe's 21 hand landmarks for better tracking.
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)
    if not res.multi_hand_landmarks:
        return None

    # Use first detected hand
    lm = res.multi_hand_landmarks[0]

    # Convert all 21 landmarks to pixel coordinates
    landmarks = []
    for landmark in lm.landmark:
        x = landmark.x * w
        y = landmark.y * h
        landmarks.append((x, y))

    # Calculate hand center using wrist (landmark 0) and palm center
    # Palm center is average of MCP joints (landmarks 5, 9, 13, 17)
    wrist = landmarks[0]
    mcp_joints = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]

    palm_center_x = (wrist[0] + sum(joint[0] for joint in mcp_joints)) / 5
    palm_center_y = (wrist[1] + sum(joint[1] for joint in mcp_joints)) / 5

    return palm_center_x, palm_center_y, landmarks


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run program."""
    args = parse_args()
    cap = open_source(args.source)

    differ_volume: float = 0.03
    announce_volume_t: float = 0.3
    frame_diff: int = 20
    seek_position_threshold: float = 0.05
    break_key: int = 27
    low_fps: float = 30

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    # Choose tracking method
    if args.no_kalman:
        tracker = SimpleSmoother(window_size=args.smooth_frames)
    else:
        tracker = ConstantAccelerationKalman(dt=1/120.0, process_var=args.process_var, meas_var=args.meas_var)

    # Gesture state
    is_playing = False
    last_play_toggle_frame = -9999
    seek_position_accum = 0.0
    volume_level = 0.5
    last_volume_shown = volume_level
    last_volume_announce_t = 0.0
    command_log = []  # list of tuples: (text, expire_time)
    hand_detected = False

    frame_idx = 0
    fps_t0 = time.time()
    fps_frames = 0
    last_frame_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, _ = gray.shape
        vis = frame.copy()

        # Measure hand landmarks
        now_time = time.time()
        dt = max(1e-3, now_time - last_frame_time)
        last_frame_time = now_time
        hand_data = get_hand_landmarks(hands, frame)
        if hand_data is not None:
            cx, cy, landmarks = hand_data
            if args.no_kalman:
                tracker.update(cx, cy, dt=dt)
            else:
                tracker.update(np.array([cx, cy], dtype=np.float32), r_scale=1.0)
            hand_detected = True
        else:
            if not args.no_kalman:
                tracker.predict()
            hand_detected = False
            landmarks = None

        # Velocity
        vx, vy = tracker.get_velocity()

        # Gestures
        if hand_detected:
            # 1) Raise hand to toggle play/pause (debounced by 20 frames)
            _, ky = tracker.get_position()
            rel_y = ky / max(1.0, float(h))
            raised = rel_y < args.raise_threshold

            if raised and (frame_idx - last_play_toggle_frame) > frame_diff:
                is_playing = not is_playing
                last_play_toggle_frame = frame_idx
                # Log command overlay (2 seconds)
                command_log.append(("PLAY" if is_playing else "PAUSE", time.time() + 2.0))

            # 2) Horizontal movement -> seek delta seconds (only if velocity is significant)
            if abs(vx) > args.min_velocity:
                # vx is px/s; convert to seconds change using sensitivity (s per px)
                seek_delta = vx * args.seek_sensitivity * dt
                seek_position_accum += seek_delta

                if abs(seek_position_accum) > seek_position_threshold:  # announce when >50ms
                    # Log command overlay (1.5 seconds)
                    sign = "+" if seek_position_accum >= 0 else ""
                    command_log.append((f"REWIND {sign}{seek_position_accum:.2f}s", time.time() + 1.5))
                    seek_position_accum = 0.0

            # 3) Vertical movement -> volume change (only if velocity is significant)
            if abs(vy) > args.min_velocity:
                # vy is px/s; convert to volume delta using sensitivity (volume per px)
                volume_level = float(np.clip(volume_level - vy * args.volume_sensitivity * dt, 0.0, 1.0))
                # Announce volume occasionally to screen (rate-limited)
                now_t = time.time()
                if abs(volume_level - last_volume_shown) >= differ_volume and \
                   (now_t - last_volume_announce_t) >= announce_volume_t:
                    command_log.append((f"VOLUME {volume_level:.2f}", now_t + 1.0))
                    last_volume_shown = volume_level
                    last_volume_announce_t = now_t
        elif len(command_log) == 0 or command_log[-1][0] != "NO SIGNAL":
            # No hand detected
            command_log.append(("NO SIGNAL", time.time() + 1.0))

        # Visualization
        if args.display:
            if hand_detected and landmarks is not None:
                # Draw all 21 hand landmarks
                for i, (lx, ly) in enumerate(landmarks):
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Wrist in green, others in red
                    cv2.circle(vis, (int(lx), int(ly)), 3, color, -1)

                # Draw hand center (tracked position)
                cx, cy = tracker.get_position()
                cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 255), 2)

                # Draw connections between landmarks (optional - shows hand structure)
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17),  # Palm connections
                ]
                for start_idx, end_idx in connections:
                    start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                    end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                    cv2.line(vis, start_point, end_point, (128, 128, 128), 1)

            # Status text
            status_text = f"Hand: {'DETECTED' if hand_detected else 'NOT DETECTED'}"
            if hand_detected:
                status_text += f" | Play: {'ON' if is_playing else 'OFF'} | Volume: {volume_level:.2f}"
            cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Velocity info
            if hand_detected:
                velocity_text = f"Velocity: vx={vx:6.1f} vy={vy:6.1f}"
                cv2.putText(vis, velocity_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Prune expired commands
            now_t = time.time()
            command_log = [c for c in command_log if c[1] > now_t]

            # Draw recent commands stacked
            y0 = 100
            for i, (msg, _) in enumerate(command_log[-6:]):
                y = y0 + i * 30
                # shadow
                cv2.putText(vis, msg, (11, y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
                # main text
                color = (0,255,255) if msg != "NO SIGNAL" else (0,0,255)
                cv2.putText(vis, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("hand_gestures", vis)
            if (cv2.waitKey(1) & 0xFF) == break_key:
                break

        frame_idx += 1
        fps_frames += 1
        if fps_frames == low_fps:
            t1 = time.time()
            fps = fps_frames / (t1 - fps_t0)
            fps_t0 = t1
            fps_frames = 0
            if not args.no_kalman:
                tracker.dt = max(1e-3, 1.0 / max(1.0, fps))
                tracker.update_F_H()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
