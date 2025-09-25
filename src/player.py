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

from src.audio.worker import AudioWorker
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

        # Seek bar + loop markers
        seek_container = ttk.Frame(self)
        seek_container.pack(fill=tk.X, padx=10, pady=8)

        self.position_var = tk.DoubleVar(value=0.0)
        self.scale = ttk.Scale(
            seek_container,
            from_=0.0,
            to=1000.0,
            orient=tk.HORIZONTAL,
            variable=self.position_var,
            command=self.on_seek,
        )
        self.scale.grid(row=0, column=0, sticky="ew")

        loop_bar = ttk.Frame(seek_container)
        loop_bar.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        loop_bar.columnconfigure(1, weight=1)

        ttk.Label(loop_bar, text="Loop A:").grid(row=0, column=0, sticky="w")
        self.loop_a_var = tk.StringVar(value="--")
        self.loop_a_label = ttk.Label(loop_bar, width=10, textvariable=self.loop_a_var)
        self.loop_a_label.grid(row=0, column=1, sticky="w")

        ttk.Label(loop_bar, text="Loop B:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.loop_b_var = tk.StringVar(value="--")
        self.loop_b_label = ttk.Label(loop_bar, width=10, textvariable=self.loop_b_var)
        self.loop_b_label.grid(row=0, column=3, sticky="w")

        ttk.Label(loop_bar, text="State:").grid(row=0, column=4, sticky="w", padx=(12, 0))
        self.loop_state_var = tk.StringVar(value="Off")
        self.loop_status_var = tk.StringVar(value="Inactive")

        loop_state = ttk.Label(loop_bar, textvariable=self.loop_state_var, width=12)
        loop_state.grid(row=0, column=5, sticky="w", padx=(12, 0))

        loop_status = ttk.Label(loop_bar, textvariable=self.loop_status_var, width=12)
        loop_status.grid(row=0, column=6, sticky="w")

        seek_container.columnconfigure(0, weight=1)

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

        # Audio Recognition for play/pause
        audio_frame = ttk.Frame(self)
        audio_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.audio_enabled = tk.BooleanVar(value=False)
        self.btn_audio = ttk.Checkbutton(audio_frame, text="Enable Audio Recognition",
                                         variable=self.audio_enabled,
                                         command=self.on_toggle_audio)
        self.btn_audio.grid(row=0, column=0, padx=4, sticky="w")
        ttk.Label(audio_frame, text="Audio: 4 hits=start music, 2 hits=stop music").grid(row=0, column=1, sticky="w")
        # Status label (Active/Disabled)
        self.audio_status_var = tk.StringVar(value="Audio Recognition: Disabled")
        self.audio_status_label = ttk.Label(audio_frame, textvariable=self.audio_status_var)
        self.audio_status_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 0))

        # Gestures for seek and volume
        gesture_frame = ttk.Frame(self)
        gesture_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.gesture_enabled = tk.BooleanVar(value=False)
        self.btn_gesture = ttk.Checkbutton(gesture_frame, text="Enable Gestures",
                                           variable=self.gesture_enabled,
                                           command=self.on_toggle_gestures)
        self.btn_gesture.grid(row=0, column=0, padx=4, sticky="w")
        ttk.Label(gesture_frame, text="Gestures: horizontal=seek, vertical=volume, V-sign=toggle active").grid(
            row=0, column=1, sticky="w")
        # Status label (Active/Paused/Disabled)
        self.gesture_status_var = tk.StringVar(value="Gestures: Disabled")
        self.gesture_status_label = ttk.Label(gesture_frame, textvariable=self.gesture_status_var)
        self.gesture_status_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 0))

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

        # Thread-safe communication queues
        self._audio_queue = queue.Queue()
        self._gesture_queue = queue.Queue()

        # Bind seek interactions to avoid fighting with programmatic updates
        self.scale.bind("<ButtonPress-1>", self._on_seek_start)
        self.scale.bind("<ButtonRelease-1>", self._on_seek_end)

        # Start UI update loop using Tk's event loop (thread-safe)
        self.after(50, self._update_ui)
        # Start audio processing loop
        self.after(10, self._process_audio_queue)
        # Start gesture processing loop
        self.after(10, self._process_gesture_queue)

        # Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

        # Worker handles
        self._audio_worker: AudioWorker | None = None
        self._gesture_worker: GestureWorker | None = None

        # Loop A/B state (milliseconds)
        self._loop_a_ms: int | None = None
        self._loop_b_ms: int | None = None
        self._loop_enabled: bool = False

        self._update_loop_labels()

    def destroy(self) -> None:
        """Destroy the music player."""
        self._running = False
        try:
            self.player.stop()
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not stop player: %s", e)
        # Stop workers if running
        try:
            if self._audio_worker is not None:
                self._audio_worker.stop()
        except Exception:
            logger.exception("Could not stop audio worker")

        try:
            if self._gesture_worker is not None:
                self._gesture_worker.stop()
        except Exception:
            logger.exception("Could not stop gesture worker")

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
            # Sync audio worker state
            if self._audio_worker is not None:
                self._audio_worker.sync_music_state(is_playing=True)
            # While audio is playing, fully stop gesture recognition
            self._auto_disable_gestures_for_playback()

    def pause(self) -> None:
        """Pause the current file."""
        self.player.pause()
        logger.info("Pause")
        # Sync audio worker state
        if self._audio_worker is not None:
            self._audio_worker.sync_music_state(is_playing=False)
        # When music is paused/stopped, allow gestures again if enabled
        self._auto_enable_gestures_if_allowed()

    def stop(self) -> None:
        """Stop the current file."""
        self.player.stop()
        logger.info("Stop")
        # Sync audio worker state
        if self._audio_worker is not None:
            self._audio_worker.sync_music_state(is_playing=False)
        # When music is stopped, allow gestures again if enabled
        self._auto_enable_gestures_if_allowed()

    def _auto_disable_gestures_for_playback(self) -> None:
        """Disable gesture recognition while music is playing."""
        try:
            if self._gesture_worker is not None:
                self._gesture_worker.stop()
                self._gesture_worker = None
                self.gesture_status_var.set("Gestures: Disabled (music playing)")
            # Also stop viewer if running
            proc = getattr(self, "_viewer_proc", None)
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    logger.exception("Could not stop viewer")
                self._viewer_proc = None
        except Exception:
            logger.exception("Failed to disable gestures for playback")

    def _auto_enable_gestures_if_allowed(self) -> None:
        """Enable gesture recognition if user toggled it on and nothing is playing."""
        try:
            is_playing = bool(self.player.is_playing())
            if is_playing:
                return
            if self.gesture_enabled.get() and self._gesture_worker is None:
                logger.info("Auto-starting gesture worker (music not playing)")
                self._gesture_worker = GestureWorker(
                    on_toggle_play=self._gesture_toggle_play,
                    on_seek_delta=self._gesture_seek_delta,
                    on_volume_delta=self._gesture_volume_delta,
                    on_loop_set_a=self._gesture_loop_set_a,
                    on_loop_set_b=self._gesture_loop_set_b,
                    on_loop_clear=self._gesture_loop_clear,
                    on_status_change=self._gesture_status_changed,
                    show_window=bool(self.show_tracking.get()),
                    initial_paused=True,
                )
                self._gesture_worker.start()
                self.gesture_status_var.set("Gestures: Paused")
        except Exception:
            logger.exception("Failed to enable gestures after stop/pause")

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

    def on_toggle_audio(self) -> None:
        """Enable or disable audio recognition control."""
        enabled = bool(self.audio_enabled.get())
        if enabled and self._audio_worker is None:
            logger.info("Starting audio worker...")
            self._audio_worker = AudioWorker(
                on_start_music=self._audio_start_music,
                on_stop_music=self._audio_stop_music,
                on_status_change=self._audio_status_changed,
            )
            self._audio_worker.start()
            # Status will be set by the worker's on_status_change callback
            logger.info("Audio recognition enabled")
        elif not enabled and self._audio_worker is not None:
            self._audio_worker.stop()
            self._audio_worker = None
            self.audio_status_var.set("Audio Recognition: Disabled")
            logger.info("Audio recognition disabled")

    def on_toggle_gestures(self) -> None:
        """Enable or disable gesture control."""
        enabled = bool(self.gesture_enabled.get())
        if enabled and self._gesture_worker is None:
            logger.info("Starting gesture worker...")
            self._gesture_worker = GestureWorker(
                on_toggle_play=self._gesture_toggle_play,
                on_seek_delta=self._gesture_seek_delta,
                on_volume_delta=self._gesture_volume_delta,
                on_loop_set_a=self._gesture_loop_set_a,
                on_loop_set_b=self._gesture_loop_set_b,
                on_loop_clear=self._gesture_loop_clear,
                on_status_change=self._gesture_status_changed,
                show_window=bool(self.show_tracking.get()),
                initial_paused=True,
            )
            # If music is currently playing, do not start; mark disabled-by-playback
            if self.player.is_playing():
                logger.info("Music is playing; gestures will remain disabled until playback stops")
                self._gesture_worker = None
                self.gesture_status_var.set("Gestures: Disabled (music playing)")
            else:
                self._gesture_worker.start()
                self.gesture_status_var.set("Gestures: Paused")
                logger.info("Gestures enabled")
        elif not enabled and self._gesture_worker is not None:
            self._gesture_worker.stop()
            self._gesture_worker = None
            self.gesture_status_var.set("Gestures: Disabled")
            logger.info("Gestures disabled")

        # Also (re)configure auto-enable worker template to include loop callbacks

    def on_tracking_toggle(self) -> None:
        """Launch or stop external tracking viewer process."""
        if bool(self.show_tracking.get()):
            try:
                self._viewer_proc = subprocess.Popen([sys.executable, "-m", "src.gestures.viewer"], close_fds=True)  # noqa: S603
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

    def on_calibrate(self) -> None:
        """Run calibration in a separate process to avoid GUI conflicts."""
        try:
            subprocess.Popen([sys.executable, "-m", "src.calibration"], close_fds=True)  # noqa: S603
            logger.info("Calibration launched in separate process")
        except Exception:
            logger.exception("Failed to launch calibration process")

    def _audio_start_music(self) -> None:
        """Start music from audio recognition in the Tk thread."""
        logger.info("Audio recognition: starting music")
        self._audio_queue.put(("start_music", None))

    def _audio_stop_music(self) -> None:
        """Stop music from audio recognition in the Tk thread."""
        logger.info("Audio recognition: stopping music")
        self._audio_queue.put(("stop_music", None))

    def _audio_status_changed(self, is_active: bool) -> None:  # noqa: FBT001
        """Receive audio recognition active status from worker (thread-safe)."""
        self._audio_queue.put(("status", is_active))

    def _gesture_toggle_play(self) -> None:
        """Toggle play/pause from gesture in the Tk thread."""
        logger.info("Gesture toggle play")
        self._gesture_queue.put(("toggle_play", None))

    def _gesture_seek_delta(self, delta_seconds: float) -> None:
        """Apply seek delta from gesture in the Tk thread."""
        logger.info("Gesture seek delta: %s", delta_seconds)
        self._gesture_queue.put(("seek_delta", delta_seconds))

    def _gesture_volume_delta(self, delta_volume: float) -> None:
        """Apply volume delta (0..100 scale) from gesture in the Tk thread."""
        logger.info("Gesture volume delta: %s", delta_volume)
        # Put the volume delta in the queue for the main thread to process
        self._gesture_queue.put(("volume_delta", delta_volume))

    def _gesture_status_changed(self, is_active: bool) -> None:  # noqa: FBT001
        """Receive gesture active/paused status from worker (thread-safe)."""
        self._gesture_queue.put(("status", is_active))

    def _on_seek_start(self, _event: tk.Event) -> None:
        """Mark that the user started dragging the seek bar."""
        self._user_seeking = True

    def _on_seek_end(self, _event: tk.Event) -> None:
        """User released the seek bar; apply final seek and resume updates."""
        try:
            self.on_seek("")
        finally:
            self._user_seeking = False

    def _process_audio_queue(self) -> None:
        """Process queued audio commands in the main Tk thread."""
        if not self._running:
            return
        try:
            while True:
                if self._audio_queue.empty():
                    break

                command, value = self._audio_queue.get_nowait()
                logger.info("Processing audio command: %s, value: %s", command, value)

                if command == "start_music":
                    self._do_start_music()
                elif command == "stop_music":
                    self._do_stop_music()
                elif command == "status":
                    # value True => active, False => inactive
                    self.audio_status_var.set("Audio Recognition: Active" if value else "Audio Recognition: Inactive")
                    logger.info("Audio status updated: %s", "Active" if value else "Inactive")

        except Exception:
            logger.exception("Error processing audio queue")
        finally:
            # Schedule next check
            self.after(10, self._process_audio_queue)

    def _process_gesture_queue(self) -> None:
        """Process queued gesture commands in the main Tk thread."""
        if not self._running:
            return
        try:
            while True:
                if self._gesture_queue.empty():
                    break

                command, value = self._gesture_queue.get_nowait()
                logger.info("Processing gesture command: %s, value: %s", command, value)

                if command == "volume_delta":
                    self._do_volume_delta(value)
                elif command == "seek_delta":
                    self._do_seek_delta(value)
                elif command == "toggle_play":
                    # Ignore play/pause from gestures while audio logic handles it
                    # But if music is not playing, let it toggle via UI play/pause
                    if not self.player.is_playing():
                        self._do_toggle_play()
                elif command == "status":
                    # value True => active, False => paused
                    self.gesture_status_var.set("Gestures: Active" if value else "Gestures: Paused")
                    logger.info("Gesture status updated: %s", "Active" if value else "Paused")
                elif command == "loop_set_a":
                    self._do_loop_set_a()
                elif command == "loop_set_b":
                    self._do_loop_set_b()
                elif command == "loop_clear":
                    self._do_loop_clear()

        except Exception:
            logger.exception("Error processing gesture queue")
        finally:
            # Schedule next check
            self.after(10, self._process_gesture_queue)

    def _do_start_music(self) -> None:
        """Start music in the main Tk thread."""
        try:
            if not self.player.is_playing():
                self.play()
                logger.info("Music started by audio recognition")
        except Exception:
            logger.exception("Error in _do_start_music")

    def _do_stop_music(self) -> None:
        """Stop music in the main Tk thread."""
        try:
            # Use pause instead of stop so playback can resume from the same position
            state = self.player.get_state()
            if state is not None and state != vlc.State.NothingSpecial:
                self.pause()
                logger.info("Music paused by audio recognition")
        except Exception:
            logger.exception("Error in _do_stop_music")

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

    def _gesture_loop_set_a(self) -> None:
        """Queue: set loop point A to current position."""
        self._gesture_queue.put(("loop_set_a", None))

    def _gesture_loop_set_b(self) -> None:
        """Queue: set loop point B to current position."""
        self._gesture_queue.put(("loop_set_b", None))

    def _gesture_loop_clear(self) -> None:
        """Queue: clear loop points."""
        self._gesture_queue.put(("loop_clear", None))

    def _get_current_time_ms(self) -> int | None:
        try:
            t = self.player.get_time()
            if t is None or t < 0:
                pos = self.player.get_position() or 0.0
                length_ms = self.player.get_length() or 0
                return int(pos * max(0, length_ms)) if length_ms > 0 else None
            return int(t)
        except Exception:
            logger.exception("Could not read current time")
            return None

    def _do_loop_set_a(self) -> None:
        t = self._get_current_time_ms()
        if t is None:
            return
        self._loop_a_ms = t
        logger.info("Loop A set at %d ms", t)
        # Enable loop only when both points are valid and ordered
        if self._loop_b_ms is not None:
            if self._loop_b_ms <= self._loop_a_ms:
                # swap to keep A < B
                self._loop_a_ms, self._loop_b_ms = self._loop_b_ms, self._loop_a_ms
            self._loop_enabled = True
        self._update_loop_labels()

    def _do_loop_set_b(self) -> None:
        t = self._get_current_time_ms()
        if t is None:
            return
        self._loop_b_ms = t
        logger.info("Loop B set at %d ms", t)
        if self._loop_a_ms is not None:
            if self._loop_b_ms <= self._loop_a_ms:
                # swap to keep A < B
                self._loop_a_ms, self._loop_b_ms = self._loop_b_ms, self._loop_a_ms
            self._loop_enabled = True
        self._update_loop_labels()

    def _do_loop_clear(self) -> None:
        logger.info("Loop cleared")
        self._loop_a_ms = None
        self._loop_b_ms = None
        self._loop_enabled = False
        self._update_loop_labels()

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
                # Enforce loop if enabled
                if self._loop_enabled and self._loop_a_ms is not None and self._loop_b_ms is not None:
                    cur = self._get_current_time_ms()
                    if cur is not None and cur >= self._loop_b_ms:
                        # Jump back near A (small offset to avoid sticky boundary)
                        target = max(0, self._loop_a_ms)
                        try:
                            self.player.set_time(int(target))
                        except Exception:
                            logger.exception("Failed to loop back to A")

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
            self._update_loop_labels()
        except Exception:
            logger.exception("Error updating UI")
        finally:
            # Schedule next update
            self.after(50, self._update_ui)

    def _format_ms(self, value: int | None) -> str:
        """Format milliseconds to mm:ss.m."""
        if value is None:
            return "--"
        seconds = max(0.0, value / 1000.0)
        minutes = int(seconds // 60)
        remainder = seconds % 60
        return f"{minutes:02d}:{remainder:04.1f}"

    def _compute_loop_status(self) -> tuple[str, str]:
        if self._loop_enabled and self._loop_a_ms is not None and self._loop_b_ms is not None:
            a = self._loop_a_ms
            b = self._loop_b_ms
            if b <= a:
                a, b = b, a
            duration_ms = max(0, b - a)
            return "Active", f"{duration_ms / 1000.0:.1f}s"
        if self._loop_a_ms is not None or self._loop_b_ms is not None:
            return "Pending", "--"
        return "Off", "Inactive"

    def _update_loop_labels(self) -> None:
        self.loop_a_var.set(self._format_ms(self._loop_a_ms))
        self.loop_b_var.set(self._format_ms(self._loop_b_ms))
        state_text, status_text = self._compute_loop_status()
        self.loop_state_var.set(state_text)
        self.loop_status_var.set(status_text)




def main() -> None:
    """Run main function."""
    app = MusicPlayer()
    app.mainloop()


if __name__ == "__main__":
    main()


