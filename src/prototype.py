import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from tracking.kalman import ConstantAccelerationKalman


def parse_args():
    ap = argparse.ArgumentParser(description="Hand gesture controller (MediaPipe Hands + Kalman)")
    ap.add_argument("--source", default=0, help="Camera index or video path")
    ap.add_argument("--display", action="store_true", help="Show visualization window")
    ap.add_argument("--midi", action="store_true", help="Enable MIDI output")
    ap.add_argument("--osc", action="store_true", help="Enable OSC output")
    ap.add_argument("--osc_host", default="127.0.0.1")
    ap.add_argument("--osc_port", type=int, default=9000)
    ap.add_argument("--midi_channel", type=int, default=0)
    ap.add_argument("--seek_sensitivity", type=float, default=0.002, help="Seconds per pixel of horizontal move")
    ap.add_argument("--volume_sensitivity", type=float, default=0.004, help="Volume per pixel of vertical move")
    ap.add_argument("--raise_threshold", type=float, default=0.25, help="Relative height to consider hand raised (0..1 from top)")
    ap.add_argument("--flip", action="store_true", help="Mirror camera for selfie view")
    return ap.parse_args()


def open_source(src):
    try:
        src_int = int(src)
        cap = cv2.VideoCapture(src_int)
    except ValueError:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {src}")
    # Try to set higher FPS if webcam
    cap.set(cv2.CAP_PROP_FPS, 120)
    return cap

def _mp_load():
    # Lazy import to avoid heavy import when unused
    try:
        import mediapipe as mp
        return mp
    except Exception:
        return None

def get_hand_point(hands, frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return pixel coordinates (cx, cy) of hand reference point (wrist/palm center)."""
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)
    if not res.multi_hand_landmarks:
        return None
    # Use first detected hand
    lm = res.multi_hand_landmarks[0]
    # Compute simple palm center as average of wrist and MCP joints
    idxs = [0, 5, 9, 13, 17]
    xs = [lm.landmark[i].x for i in idxs]
    ys = [lm.landmark[i].y for i in idxs]
    cx = float(np.mean(xs) * w)
    cy = float(np.mean(ys) * h)
    return cx, cy


def main():
    args = parse_args()
    cap = open_source(args.source)

    # Outputs
    midi_out = None
    osc_out = None
    if args.midi:
        from io_out import MidiOut
        midi_out = MidiOut(channel=args.midi_channel)
    if args.osc:
        from io_out import OscOut
        osc_out = OscOut(host=args.osc_host, port=args.osc_port)

    # MediaPipe setup
    mp = _mp_load()
    if mp is None:
        raise RuntimeError("mediapipe is not installed. Please install mediapipe.")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    kalman = ConstantAccelerationKalman(dt=1/120.0, process_var=50.0, meas_var=20.0)

    # Gesture state
    is_playing = False
    last_play_toggle_frame = -9999
    seek_position_accum = 0.0
    volume_level = 0.5
    last_volume_shown = volume_level
    last_volume_announce_t = 0.0
    command_log = []  # list of tuples: (text, expire_time)

    # Flow/KLT not used in hand tracking path; Kalman smooths landmark positions
    frame_idx = 0
    fps_t0 = time.time()
    fps_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        vis = frame.copy()

        # Measure hand point
        pt = get_hand_point(hands, frame)
        if pt is not None:
            cx, cy = pt
            kalman.update(np.array([cx, cy], dtype=np.float32), r_scale=1.0)
        else:
            kalman.predict()

        # Velocity
        vx, vy = kalman.get_velocity()

        # Gestures
        # 1) Raise hand to toggle play/pause (debounced by 20 frames)
        kx, ky = kalman.get_position()
        rel_y = ky / max(1.0, float(h))
        raised = rel_y < args.raise_threshold
        if raised and (frame_idx - last_play_toggle_frame) > 20:
            is_playing = not is_playing
            last_play_toggle_frame = frame_idx
            if osc_out is not None:
                osc_out.send("/transport", 1 if is_playing else 0)
            # Log command overlay (2 seconds)
            command_log.append(("PLAY" if is_playing else "PAUSE", time.time() + 2.0))
        # 2) Horizontal movement -> seek delta seconds (send as OSC)
        seek_delta = vx * args.seek_sensitivity
        seek_position_accum += seek_delta
        if abs(seek_position_accum) > 0.05:  # send when >50ms
            if osc_out is not None:
                osc_out.send("/seek_delta", float(seek_position_accum))
            # Log command overlay (1.5 seconds)
            sign = "+" if seek_position_accum >= 0 else ""
            command_log.append((f"SEEK {sign}{seek_position_accum:.2f}s", time.time() + 1.5))
            seek_position_accum = 0.0
        # 3) Vertical movement -> volume change
        volume_level = float(np.clip(volume_level - vy * args.volume_sensitivity, 0.0, 1.0))
        if osc_out is not None:
            osc_out.send("/volume", float(volume_level))
        if midi_out is not None:
            midi_out.cc(7, int(volume_level * 127))
        # Announce volume occasionally to screen (rate-limited)
        now_t = time.time()
        if abs(volume_level - last_volume_shown) >= 0.03 and (now_t - last_volume_announce_t) >= 0.3:
            command_log.append((f"VOLUME {volume_level:.2f}", now_t + 1.0))
            last_volume_shown = volume_level
            last_volume_announce_t = now_t

        # Visualization
        if args.display:
            cx, cy = kalman.get_position()
            cv2.circle(vis, (int(cx), int(cy)), 6, (0,255,0), -1)
            txt1 = f"play={'ON' if is_playing else 'OFF'} vol={volume_level:.2f}"
            txt2 = f"vx={vx:6.1f} vy={vy:6.1f} raised={raised}"
            cv2.putText(vis, txt1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(vis, txt2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            # Prune expired commands
            now_t = time.time()
            command_log = [c for c in command_log if c[1] > now_t]
            # Draw recent commands stacked
            y0 = 90
            for i, (msg, expire) in enumerate(command_log[-6:]):
                y = y0 + i * 24
                # shadow
                cv2.putText(vis, msg, (11, y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(vis, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("hand_gestures", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        frame_idx += 1
        fps_frames += 1
        if fps_frames == 30:
            t1 = time.time()
            fps = fps_frames / (t1 - fps_t0)
            fps_t0 = t1
            fps_frames = 0
            kalman.dt = max(1e-3, 1.0 / max(1.0, fps))
            kalman._update_F_H()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
