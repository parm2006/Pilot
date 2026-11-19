"""Local Whisper dictation from the same 16 kHz PCM stream as voice commands."""
from collections import deque

import numpy as np

from pilot_paths import cached_data

SAMPLE_RATE = 16000
FRAME_BYTES = 640  # WebRTC VAD accepts 20 ms of mono int16 audio at 16 kHz.


class SpeechBuffer:
    """Capture one phrase, finishing after a pause or returning b'' after no speech."""

    def __init__(self, gap=0.8, vad=None):
        if not 0.2 <= gap <= 5.0:
            raise ValueError("dictation gap must be between 0.2 and 5 seconds")
        if vad is None:
            import webrtcvad
            vad = webrtcvad.Vad(2)
        self.vad = vad
        self.gap_frames = round(gap / 0.02)
        self.pending = b""
        self.lead = deque(maxlen=10)
        self.frames = []
        self.speech_frames = 0
        self.silent_frames = 0
        self.total_frames = 0

    def feed(self, data):
        self.pending += data
        while len(self.pending) >= FRAME_BYTES:
            frame, self.pending = self.pending[:FRAME_BYTES], self.pending[FRAME_BYTES:]
            speech = self.vad.is_speech(frame, SAMPLE_RATE)
            self.total_frames += 1
            if speech:
                if not self.frames:
                    self.frames.extend(self.lead)
                    self.lead.clear()
                self.frames.append(frame)
                self.speech_frames += 1
                self.silent_frames = 0
            elif self.frames:
                self.frames.append(frame)
                self.silent_frames += 1
            else:
                self.lead.append(frame)
            if self.frames and self.silent_frames >= self.gap_frames:
                return b"".join(self.frames) if self.speech_frames >= 6 else b""
            if not self.frames and self.total_frames >= 500:  # ten seconds to begin speaking
                return b""
            # shortcut: cap a single phrase at 60 seconds; use chunked decoding for longer dictation.
            if len(self.frames) >= 3000:
                return b"".join(self.frames)
        return None


class LocalWhisper:
    def __init__(self, model_name="base.en"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4,
                                  download_root=str(cached_data("whisper")), local_files_only=True)

    def transcribe(self, pcm):
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio, language="en", beam_size=3,
                                             condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segments if s.text.strip())


def foreground_window():
    import ctypes
    get_window = ctypes.windll.user32.GetForegroundWindow
    get_window.restype = ctypes.c_void_p
    return get_window()


def type_text(text):
    # pynput uses Windows Unicode input for characters outside the keyboard layout.
    from pynput.keyboard import Controller
    Controller().type(text)
