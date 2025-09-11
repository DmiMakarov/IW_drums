from typing import Optional

# MIDI output (optional)
try:
    import rtmidi
except Exception:
    rtmidi = None

# OSC output (optional)
try:
    from pythonosc.udp_client import SimpleUDPClient
except Exception:
    SimpleUDPClient = None

class MidiOut:
    def __init__(self, port_name: str = "StickTracker", channel: int = 0):
        self.channel = max(0, min(channel, 15))
        self.midiout = None
        if rtmidi is not None:
            try:
                self.midiout = rtmidi.MidiOut()
                self.midiout.open_virtual_port(port_name)
            except Exception:
                self.midiout = None

    def note_on(self, note: int = 60, velocity: int = 100):
        if self.midiout is None:
            return
        status = 0x90 | self.channel
        self.midiout.send_message([status, int(note) & 0x7F, int(velocity) & 0x7F])

    def note_off(self, note: int = 60):
        if self.midiout is None:
            return
        status = 0x80 | self.channel
        self.midiout.send_message([status, int(note) & 0x7F, 0])

    def cc(self, cc_num: int, value: int):
        if self.midiout is None:
            return
        status = 0xB0 | self.channel
        self.midiout.send_message([status, int(cc_num) & 0x7F, int(value) & 0x7F])

class OscOut:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.client = None
        if SimpleUDPClient is not None:
            try:
                self.client = SimpleUDPClient(host, int(port))
            except Exception:
                self.client = None

    def send(self, address: str = "/hit", *args):
        if self.client is None:
            return
        try:
            self.client.send_message(address, args if len(args) != 1 else args[0])
        except Exception:
            pass
