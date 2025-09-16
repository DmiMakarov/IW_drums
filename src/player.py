"""Simple Tkinter-based music player using VLC backend."""
from __future__ import annotations

import logging
import queue
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import vlc

from src.gestures.worker import GestureWorker

# Optional heavy deps are imported lazily inside the gesture worker

logger = logging.getLogger(__name__)


class MusicPlayer(tk.Tk):
    """Simple Tkinter-based music player using VLC backend."""

    def __init__(self) -> None:
        """Initialize the music player."""
        super().__init__()
        self.title("IW Drums - Player")
        self.geometry("520x200")

        # VLC
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.media: vlc.Media | None = None

        # UI
        self.file_label = ttk.Label(self, text="No file loaded")
        self.file_label.pack(pady=6)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8)

        self.btn_open = ttk.Button(controls, text="Open", command=self.open_file)
        self.btn_open.grid(row=0, column=0, padx=4)

        self.btn_play = ttk.Button(controls, text="Play", command=self.play)
        self.btn_play.grid(row=0, column=1, padx=4)

        self.btn_pause = ttk.Button(controls, text="Pause", command=self.pause)
        self.btn_pause.grid(row=0, column=2, padx=4)

        self.btn_stop = ttk.Button(controls, text="Stop", command=self.stop)
        self.btn_stop.grid(row=0, column=3, padx=4)

        # Seek bar
        self.position_var = tk.DoubleVar(value=0.0)
        self.scale = ttk.Scale(self, from_=0.0, to=1000.0, orient=tk.HORIZONTAL,
                               variable=self.position_var, command=self.on_seek)
        self.scale.pack(fill=tk.X, padx=10, pady=8)

        # Volume
        vol_frame = ttk.Frame(self)
        vol_frame.pack(fill=tk.X, padx=8)
        ttk.Label(vol_frame, text="Volume").grid(row=0, column=0, padx=4)
        self.volume_var = tk.IntVar(value=80)
        self.volume_scale = ttk.Scale(vol_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                      variable=self.volume_var, command=self.on_volume)
        self.volume_scale.grid(row=0, column=1, sticky="ew", padx=4)
        # Set initial volume
        self.player.audio_set_volume(80)
        vol_frame.columnconfigure(1, weight=1)

        # Gestures & tools
        gesture_frame = ttk.Frame(self)
        gesture_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.gesture_enabled = tk.BooleanVar(value=False)
        self.btn_gesture = ttk.Checkbutton(gesture_frame, text="Enable Gestures",
                                           variable=self.gesture_enabled,
                                           command=self.on_toggle_gestures)
        self.btn_gesture.grid(row=0, column=0, padx=4, sticky="w")
        ttk.Label(gesture_frame, text="Gestures: raise=play/pause").grid(row=0, column=1, sticky="w")

        self.show_tracking = tk.BooleanVar(value=False)
        self.btn_tracking = ttk.Checkbutton(gesture_frame, text="Show Tracking",
                                            variable=self.show_tracking,
                                            command=self.on_tracking_toggle)
        self.btn_tracking.grid(row=0, column=2, padx=8, sticky="w")

        self.btn_calibrate = ttk.Button(gesture_frame, text="Calibrate", command=self.on_calibrate)
        self.btn_calibrate.grid(row=0, column=3, padx=8, sticky="e")

        # Position update state
        self._running = True
        self._user_seeking = False

        # Thread-safe communication queue
        self._gesture_queue = queue.Queue()

        # Bind seek interactions to avoid fighting with programmatic updates
        self.scale.bind("<ButtonPress-1>", self._on_seek_start)
        self.scale.bind("<ButtonRelease-1>", self._on_seek_end)

        # Start UI update loop using Tk's event loop (thread-safe)
        self.after(50, self._update_ui)
        # Start gesture processing loop
        self.after(10, self._process_gesture_queue)

        # Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

        # Gesture worker handle
        self._gesture_worker: GestureWorker | None = None

    def destroy(self) -> None:
        """Destroy the music player."""
        self._running = False
        try:
            self.player.stop()
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not stop player: %s", e)
        # Stop viewer if running
        try:
            proc = getattr(self, "_viewer_proc", None)
            if proc is not None:
                proc.terminate()
        except Exception:
            logger.exception("Could not stop viewer")

        super().destroy()

    def open_file(self) -> None:
        """Open a file."""
        path = Path(filedialog.askopenfilename(title="Open audio file",
                                              filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg"),
                                                         ("All", "*.*")]))
        if not path:
            return

        logger.info("Selected file: %s", path)

        if not path.exists():
            logger.warning("Selected file does not exist: %s", path)
            return
        self.media = self.instance.media_new_path(path)
        self.player.set_media(self.media)
        self.file_label.config(text=path.name)

        logger.info("Loaded file: %s", path)

    def play(self) -> None:
        """Play the current file."""
        if self.media is None:
            self.open_file()
        if self.media is not None:
            self.player.play()
            logger.info("Play")

    def pause(self) -> None:
        """Pause the current file."""
        self.player.pause()
        logger.info("Pause")

    def stop(self) -> None:
        """Stop the current file."""
        self.player.stop()
        logger.info("Stop")

    def on_seek(self, _value: str) -> None:
        """Seek to the current position."""
        try:
            pos = float(self.position_var.get()) / 1000.0
            self.player.set_position(max(0.0, min(1.0, pos)))
        except Exception:
            logger.exception("Error seeking to position")


    def on_volume(self, _value: str) -> None:
        """Set the volume."""
        try:
            logger.info("Setting volume to: %s", _value)
            self.player.audio_set_volume(max(0, min(100, int(float(_value)))))
        except Exception:
            logger.exception("Error setting volume")

    def on_toggle_gestures(self) -> None:
        """Enable or disable gesture control."""
        enabled = bool(self.gesture_enabled.get())
        if enabled and self._gesture_worker is None:
            logger.info("Starting gesture worker...")
            self._gesture_worker = GestureWorker(
                on_toggle_play=self._gesture_toggle_play,
                on_seek_delta=self._gesture_seek_delta,
                on_volume_delta=self._gesture_volume_delta,
                show_window=bool(self.show_tracking.get()),
            )
            self._gesture_worker.start()
            logger.info("Gestures enabled")
        elif not enabled and self._gesture_worker is not None:
            self._gesture_worker.stop()
            self._gesture_worker = None
            logger.info("Gestures disabled")

    def on_tracking_toggle(self) -> None:
        """Launch or stop external tracking viewer process."""
        if bool(self.show_tracking.get()):
            try:
                self._viewer_proc = subprocess.Popen([sys.executable, "-m", "src.gestures.viewer"], close_fds=True)
                logger.info("Gesture viewer launched")
                # Auto-enable gestures so player receives updates while viewer shows overlay
                if not self.gesture_enabled.get():
                    self.gesture_enabled.set(True)
                    self.on_toggle_gestures()
            except Exception:
                logger.exception("Failed to launch gesture viewer")
        else:
            proc = getattr(self, "_viewer_proc", None)
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    logger.exception("Could not stop viewer")
                self._viewer_proc = None

    def _gesture_toggle_play(self) -> None:
        """Toggle play/pause from gesture in the Tk thread."""
        logger.info("Gesture toggle play")
        self._gesture_queue.put(('toggle_play', None))

    def on_calibrate(self) -> None:
        """Run calibration in a separate process to avoid GUI conflicts."""
        try:
            subprocess.Popen([sys.executable, "-m", "src.calibration"], close_fds=True)
            logger.info("Calibration launched in separate process")
        except Exception:
            logger.exception("Failed to launch calibration process")
    def _gesture_seek_delta(self, delta_seconds: float) -> None:
        """Apply seek delta from gesture in the Tk thread."""
        logger.info("Gesture seek delta: %s", delta_seconds)
        self._gesture_queue.put(('seek_delta', delta_seconds))

    def _gesture_volume_delta(self, delta_volume: float) -> None:
        """Apply volume delta (0..100 scale) from gesture in the Tk thread."""
        logger.info("Gesture volume delta: %s", delta_volume)
        # Put the volume delta in the queue for the main thread to process
        self._gesture_queue.put(('volume_delta', delta_volume))

    def _on_seek_start(self, _event: tk.Event) -> None:
        """Mark that the user started dragging the seek bar."""
        self._user_seeking = True

    def _on_seek_end(self, _event: tk.Event) -> None:
        """User released the seek bar; apply final seek and resume updates."""
        try:
            self.on_seek("")
        finally:
            self._user_seeking = False

    def _process_gesture_queue(self) -> None:
        """Process queued gesture commands in the main Tk thread."""
        if not self._running:
            return
        try:
            while True:
                try:
                    command, value = self._gesture_queue.get_nowait()
                    logger.info("Processing gesture command: %s, value: %s", command, value)

                    if command == 'volume_delta':
                        self._do_volume_delta(value)
                    elif command == 'seek_delta':
                        self._do_seek_delta(value)
                    elif command == 'toggle_play':
                        self._do_toggle_play()

                except queue.Empty:
                    break
        except Exception:
            logger.exception("Error processing gesture queue")
        finally:
            # Schedule next check
            self.after(10, self._process_gesture_queue)

    def _do_volume_delta(self, delta_volume: float) -> None:
        """Apply volume delta in the main Tk thread."""
        logger.info("_do_volume_delta called with: %s", delta_volume)
        try:
            cur = self.player.audio_get_volume()
            logger.info("Current volume: %s", cur)
            if cur < 0:
                cur = int(self.volume_var.get())
            new_vol = int(max(0, min(100, cur + delta_volume)))
            logger.info("Setting new volume: %s", new_vol)
            self.volume_var.set(new_vol)
            self.on_volume(str(new_vol))
        except Exception:
            logger.exception("Error in _do_volume_delta")

    def _do_seek_delta(self, delta_seconds: float) -> None:
        """Apply seek delta in the main Tk thread."""
        try:
            cur_ms = self.player.get_time()
            if cur_ms is None or cur_ms < 0:
                # Fallback to position API
                pos = self.player.get_position() or 0.0
                length_ms = self.player.get_length() or 0
                new_ms = int(max(0, min(length_ms, pos * max(0, length_ms) + delta_seconds * 1000.0)))
            else:
                length_ms = self.player.get_length() or 0
                new_ms = int(max(0, min(length_ms, cur_ms + delta_seconds * 1000.0)))
            if length_ms > 0:
                self.player.set_time(new_ms)
                # Update UI seek bar immediately
                new_pos = max(0.0, min(1.0, float(new_ms) / float(length_ms)))
                if not self._user_seeking:
                    self.position_var.set(new_pos * 1000.0)
        except Exception:
            logger.exception("Error in _do_seek_delta")

    def _do_toggle_play(self) -> None:
        """Toggle play/pause in the main Tk thread."""
        try:
            if self.player.is_playing():
                self.pause()
            else:
                self.play()
        except Exception:
            logger.exception("Error in _do_toggle_play")

    def _update_ui(self) -> None:
        """Periodic UI updater running in the Tk thread."""
        if not self._running:
            return
        try:
            state = self.player.get_state()
            if state is not None and state != vlc.State.NothingSpecial:
                pos = self.player.get_position()
                if (
                    not self._user_seeking
                    and pos is not None
                    and isinstance(pos, (int, float))
                    and pos >= 0.0
                    and pos <= 1.0
                ):
                    # Update scale position without triggering seek jitter
                    self.position_var.set(pos * 1000.0)
        except Exception:
            logger.exception("Error updating UI")
        finally:
            # Schedule next update
            self.after(50, self._update_ui)




def main() -> None:
    """Run main function."""
    app = MusicPlayer()
    app.mainloop()


if __name__ == "__main__":
    main()


