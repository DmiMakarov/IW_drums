import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

from tracking.kalman import ConstantAccelerationKalman
from tracking.hit_detector import HitDetector, HitConfig


def parse_args():
    ap = argparse.ArgumentParser(description="Drummer stick tracker (YOLO + KLT/Kalman)")
    ap.add_argument("--source", default=0, help="Camera index or video path")
    ap.add_argument("--model", default=None, help="Path to YOLO model (optional)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--detect_every", type=int, default=3, help="Run detector every N frames")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--class_name", default="stick_tip", help="Class name to filter (if model has multiple)")
    ap.add_argument("--display", action="store_true", help="Show visualization window")
    ap.add_argument("--midi", action="store_true", help="Enable MIDI output on hit")
    ap.add_argument("--osc", action="store_true", help="Enable OSC output on hit")
    ap.add_argument("--osc_host", default="127.0.0.1")
    ap.add_argument("--osc_port", type=int, default=9000)
    ap.add_argument("--midi_note", type=int, default=60)
    ap.add_argument("--midi_channel", type=int, default=0)
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


def detect_tip_yolo(model: YOLO, frame: np.ndarray, imgsz: int, conf: float, class_name: Optional[str]) -> Optional[Tuple[float,float,float]]:
    """Return (cx, cy, conf) of best tip detection, or None."""
    res = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)
    if not res:
        return None
    r = res[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    boxes = r.boxes
    if hasattr(boxes, 'cls') and class_name is not None and r.names:
        # filter by class name if provided
        keep = []
        for i, c in enumerate(boxes.cls.cpu().numpy().astype(int)):
            nm = r.names.get(c, str(c))
            if nm == class_name:
                keep.append(i)
        if keep:
            boxes = boxes[keep]
    # pick highest confidence
    confs = boxes.conf.cpu().numpy()
    i_best = int(np.argmax(confs))
    xyxy = boxes.xyxy[i_best].cpu().numpy()
    cx = float(0.5*(xyxy[0] + xyxy[2]))
    cy = float(0.5*(xyxy[1] + xyxy[3]))
    return cx, cy, float(confs[i_best])


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

    model = None
    if args.model and YOLO_AVAILABLE:
        model = YOLO(args.model)
        try:
            model.to(args.device)
        except Exception:
            pass

    lk_params = dict(winSize=(21,21), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    kalman = ConstantAccelerationKalman(dt=1/120.0, process_var=50.0, meas_var=20.0)
    hit = HitDetector(HitConfig())

    prev_gray = None
    prev_pt = None
    frame_idx = 0
    fps_t0 = time.time()
    fps_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        vis = frame.copy()
        detected = None

        # Run YOLO every N frames if available
        if model is not None and (frame_idx % args.detect_every == 0):
            try:
                det = detect_tip_yolo(model, frame, args.imgsz, args.conf, args.class_name)
            except Exception:
                det = None
            if det is not None:
                cx, cy, c = det
                kalman.update(np.array([cx, cy], dtype=np.float32), r_scale=max(1.0, 2.0 - c))
                prev_pt = np.array([[cx, cy]], dtype=np.float32).reshape(1,1,2)
                detected = (cx, cy, c)

        # Track between detections with KLT if we have a previous point
        if prev_gray is not None and prev_pt is not None:
            nxt_pt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, **lk_params)
            ok_flow = st is not None and st[0,0] == 1 and err is not None and err[0,0] < 20
            if ok_flow:
                px, py = float(nxt_pt[0,0,0]), float(nxt_pt[0,0,1])
                kalman.update(np.array([px, py], dtype=np.float32), r_scale=1.0)
                prev_pt = nxt_pt
            else:
                # If flow lost and no recent detection, rely on prediction
                kalman.predict()
        else:
            kalman.predict()

        # Get velocity for hit logic
        vx, vy = kalman.get_velocity()
        is_hit = hit.update(vy)

        # Visualization
        if args.display:
            cx, cy = kalman.get_position()
            cv2.circle(vis, (int(cx), int(cy)), 6, (0,0,255) if is_hit else (0,255,0), -1)
            if detected is not None:
                dcx, dcy, dc = detected
                cv2.circle(vis, (int(dcx), int(dcy)), 5, (255,255,0), 2)
            txt = f"vy={vy:6.1f} hit={is_hit}"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("sticks", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        # Triggers
        if is_hit:
            if midi_out is not None:
                midi_out.note_on(args.midi_note, 100)
                midi_out.note_off(args.midi_note)
            if osc_out is not None:
                osc_out.send("/hit", 1)

        prev_gray = gray
        frame_idx += 1
        fps_frames += 1
        if fps_frames == 30:
            t1 = time.time()
            fps = fps_frames / (t1 - fps_t0)
            fps_t0 = t1
            fps_frames = 0
            # Update Kalman dt if FPS changed significantly
            kalman.dt = max(1e-3, 1.0 / max(1.0, fps))
            kalman._update_F_H()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
