# Drummer Stick Tracker (YOLO + Tracking)

## Setup

- Python 3.10+ recommended
- Install deps: `pip install -r requirements.txt`

## Run

- Webcam: `python src/prototype.py --source 0`
- Video: `python src/prototype.py --source path/to/video.mp4`

## Notes
- Use 120–240 fps cameras if possible.
- For better robustness, collect a small dataset and train a tiny model (see `datasets/template`).
