#!/usr/bin/env python3
"""
pilot_voice.py - local voice commands for pilot.py (no cloud, no API, no per-use cost)
=====================================================================================

Pipeline (all on this machine):

  microphone -> Vosk (speech-to-text) -> exact phrase?  ........ run it
                                      -> "control <key>" ....... press that combo
                                      -> embedding match ....... run the closest command
                                         (Ollama embedding model, or your fine-tuned
                                          sentence-transformers model)

* Vosk runs twice per utterance at most: first limited to your command phrases (accurate, cheap),
  then as free speech for misses (or every active utterance with --voice-verbose).
* Prints command guesses, fallback transcripts, and match results, including rejected speech.
* Say "dictate", then speak and pause: local Whisper transcribes and inserts that phrase.
* "Talk to the hand": face an open palm with fingers spread toward the camera to mute audio.
* Commands live in commands.json next to pilot.py and hot-reload when you save the file.

Needs:  pip install vosk sounddevice          (+ Ollama with an embedding model for fuzzy matching:
                                                `ollama pull all-minilm`)
The Vosk model (~40 MB) and the MediaPipe hand model (~8 MB) download automatically on first use.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from pilot_dictation import LocalWhisper, SpeechBuffer, foreground_window, type_text

APP_DIR = Path.home() / ".eye_mouse"
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")
HAND_MODEL_PATH = APP_DIR / "hand_landmarker.task"
SAMPLE_RATE = 16000
BLOCK = 2000                      # samples per audio block (0.125 s)

# --------------------------------------------------------------------------- #
# text + key helpers
# --------------------------------------------------------------------------- #
BUILTINS = {"pause_listener", "resume_listener", "stop_all", "pause_pilot", "resume_pilot",
            "recenter", "dictate", "recal", "wink", "control", "input"}
ACTION_KINDS = ("keys", "text", "run", "scroll", "hscroll", "autoscroll", "mouse", "builtin")

# Names accepted in commands.json -> pyautogui key names.
KEY_ALIASES = {
    "control": "ctrl", "windows": "win", "escape": "esc", "return": "enter", "del": "delete",
    "equals": "=", "plus": "=", "minus": "-", "comma": ",", "period": ".", "slash": "/",
    "backslash": "\\", "semicolon": ";", "quote": "'", "backtick": "`", "lbracket": "[",
    "rbracket": "]", "mute": "volumemute",
}
NAMED_KEYS = {
    "ctrl", "alt", "shift", "win", "enter", "tab", "esc", "space", "backspace", "delete", "insert",
    "home", "end", "pageup", "pagedown", "up", "down", "left", "right", "capslock", "printscreen",
    "volumeup", "volumedown", "volumemute", "playpause", "nexttrack", "prevtrack",
    "=", "-", ",", ".", "/", "\\", ";", "'", "`", "[", "]",
}


def norm_text(s: str) -> str:
    s = s.lower().replace("[unk]", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_key(name: str) -> str:
    n = KEY_ALIASES.get(name.strip().lower(), name.strip().lower())
    if n in NAMED_KEYS or re.fullmatch(r"[a-z0-9]", n) or re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", n):
        return n
    raise ValueError(f"unknown key '{name}'")


# Basic "control <letter>" shortcuts ---------------------------------------- #
# Speech engines hear letter names as words ("control see" -> ctrl+c).
LETTER_WORDS = {"see": "c", "sea": "c", "bee": "b", "be": "b", "are": "r", "you": "u", "why": "y",
                "tea": "t", "eye": "i", "oh": "o", "pea": "p", "queue": "q", "cue": "q", "ex": "x",
                "dee": "d", "gee": "g", "jay": "j", "kay": "k", "el": "l", "em": "m", "en": "n",
                "ess": "s", "double you": "w", "vee": "v", "zee": "z", "zed": "z", "aitch": "h",
                "ay": "a", "eff": "f"}


CONTROL_LETTERS = {**{chr(c): chr(c) for c in range(ord("a"), ord("z") + 1)}, **LETTER_WORDS}
CONTROL_PHRASES = [f"control {letter}" for letter in CONTROL_LETTERS]


def parse_combo(text: str) -> Optional[list[str]]:
    """'control see' -> ['ctrl', 'c']; only Control plus a single letter is supported."""
    parts = norm_text(text).split(maxsplit=1)
    if len(parts) != 2 or parts[0] not in ("control", "ctrl"):
        return None
    key = CONTROL_LETTERS.get(parts[1])
    return ["ctrl", key] if key else None


# --------------------------------------------------------------------------- #
# commands.json
# --------------------------------------------------------------------------- #
@dataclass
class Command:
    phrase: str
    action: dict
    keys: list
    fuzzy: bool = True

    @property
    def builtin(self) -> Optional[str]:
        return self.action.get("builtin")


def compile_command(phrase: str, action) -> Command:
    if not isinstance(action, dict):
        raise ValueError("must be an object")
    if not any(k in action for k in ACTION_KINDS):
        raise ValueError("needs one of " + ", ".join(ACTION_KINDS))
    keys = []
    if "keys" in action:
        raw = action["keys"]
        for combo in (raw if isinstance(raw, list) else [raw]):
            keys.append([norm_key(k) for k in str(combo).split("+")])
    if "builtin" in action and action["builtin"] not in BUILTINS:
        raise ValueError(f"unknown builtin '{action['builtin']}'")
    if "mouse" in action and action["mouse"] not in ("left", "right", "middle", "double"):
        raise ValueError(f"unknown mouse action '{action['mouse']}'")
    for k in ("scroll", "hscroll", "autoscroll"):
        if k in action:
            int(action[k])
    return Command(phrase, action, keys, bool(action.get("fuzzy", True)))


def load_commands(path: Path):
    """-> (table {normalised phrase or alias: Command}, settings dict, [problems])"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[str, Command] = {}
    problems: list[str] = []
    for phrase, action in data.get("commands", {}).items():
        key = norm_text(phrase)
        if not key:
            continue
        try:
            cmd = compile_command(key, action)
        except Exception as e:  # noqa: BLE001
            problems.append(f"'{phrase}': {e}")
            continue
        table[key] = cmd
        for alias in action.get("aliases", []):
            table.setdefault(norm_text(alias), cmd)
    return table, data.get("settings", {}), problems


# --------------------------------------------------------------------------- #
# embeddings (fuzzy matching)
# --------------------------------------------------------------------------- #
def _unit_rows(E) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


class OllamaEmbedder:
    """Local Ollama server (default http://127.0.0.1:11434). `ollama pull all-minilm` first."""

    def __init__(self, model="all-minilm", host="http://127.0.0.1:11434"):
        self.model, self.host = model, host.rstrip("/")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy

    def embed(self, texts: list[str]) -> np.ndarray:
        body = json.dumps({"model": self.model, "input": texts, "keep_alive": "30m"}).encode()
        req = urllib.request.Request(self.host + "/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with self._opener.open(req, timeout=8) as r:
            return _unit_rows(json.load(r)["embeddings"])

    def __str__(self):
        return f"ollama:{self.model}"


class STEmbedder:
    """sentence-transformers model on disk or the Hugging Face hub - e.g. your fine-tuned weights."""

    def __init__(self, name_or_path: str):
        from sentence_transformers import SentenceTransformer
        self.name = name_or_path
        self.model = SentenceTransformer(name_or_path)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _unit_rows(self.model.encode(texts, normalize_embeddings=True))

    def __str__(self):
        return f"sentence-transformers:{self.name}"


def make_embedder(backend: str, model: str):
    if backend == "none":
        return None
    if backend == "st":
        return STEmbedder(model)
    return OllamaEmbedder(model)


# --------------------------------------------------------------------------- #
# routing: text -> command
# --------------------------------------------------------------------------- #
@dataclass
class Resolved:
    label: str
    command: Command
    how: str            # exact | combo | fuzzy
    score: float = 1.0


class Router:
    def __init__(self, table, embedder=None, min_sim=0.60, margin=0.04, max_words=6, verbose=False):
        self.embedder, self.min_sim, self.margin = embedder, min_sim, margin
        self.max_words, self.verbose = max_words, verbose
        self.table: dict[str, Command] = {}
        self._index = None
        self.index_state = "off" if embedder is None else "building"
        self._warned = False
        self.set_commands(table)

    def set_commands(self, table):
        self.table = table
        self._index = None
        if self.embedder is not None:
            self.index_state = "building"
            threading.Thread(target=self._build_index, args=(table,), daemon=True).start()

    def _build_index(self, table):
        try:
            keys = [p for p, c in table.items() if c.fuzzy]
            M = np.vstack([self.embedder.embed(keys[i:i + 64]) for i in range(0, len(keys), 64)])
            if table is self.table:                      # ignore if the table was reloaded meanwhile
                self._index = (keys, M, [table[k].phrase for k in keys])
                self.index_state = "ready"
        except Exception as e:  # noqa: BLE001
            self.index_state = f"unavailable ({e})"
            print(f"[voice] fuzzy matching off: {e}")

    def route(self, text: str) -> Optional[Resolved]:
        t = norm_text(text)
        if not t:
            return None
        cmd = self.table.get(t)
        if cmd is not None:
            return Resolved(cmd.phrase, cmd, "exact")
        combo = parse_combo(t)
        if combo:
            try:
                keys = [norm_key(k) for k in combo]
            except ValueError:
                keys = None
            if keys:
                label = "+".join(keys)
                return Resolved(label, Command(label, {"keys": label}, [keys], False), "combo")
        if t.split()[0] in ("control", "ctrl", "alt", "shift", "windows", "win"):
            return None
        return self.fuzzy(t)

    def fuzzy(self, t: str) -> Optional[Resolved]:
        if self.embedder is None or self._index is None:
            return None
        if not 1 <= len(t.split()) <= self.max_words:    # long speech is conversation, not a command
            return None
        try:
            v = self.embedder.embed([t])[0]
        except Exception as e:  # noqa: BLE001
            if not self._warned:
                self._warned = True
                print(f"[voice] embedding call failed: {e}")
            return None
        keys, M, owners = self._index
        sims = M @ v
        order = np.argsort(-sims)
        best = int(order[0])
        second = next((float(sims[j]) for j in order[1:] if owners[j] != owners[best]), -1.0)
        if self.verbose:
            top = ", ".join(f"{keys[j]} {sims[j]:.2f}" for j in order[:3])
            print(f"[voice] fuzzy '{t}': {top}")
        score = float(sims[best])
        cmd = self.table.get(keys[best])
        if cmd is not None and score >= self.min_sim and score - second >= self.margin:
            return Resolved(cmd.phrase, cmd, "fuzzy", score)
        return None


# --------------------------------------------------------------------------- #
# doing things
# --------------------------------------------------------------------------- #
class Actuator:
    """Runs a Command with pyautogui. Builtins are returned to the caller instead."""

    AUTOSCROLL_EVERY = 0.25

    def __init__(self, pg, dry=False):
        self.pg, self.dry = pg, dry
        self.auto = 0
        self._last_auto = 0.0

    def _wheel(self, notches: int, horizontal=False):
        if self.dry or notches == 0:
            return
        if sys.platform.startswith("win"):
            import ctypes
            flag = 0x1000 if horizontal else 0x0800                 # HWHEEL / WHEEL
            step = 120 if notches > 0 else -120
            for _ in range(abs(notches)):
                ctypes.windll.user32.mouse_event(flag, 0, 0, step, 0)
                time.sleep(0.012)
        else:
            (self.pg.hscroll if horizontal else self.pg.scroll)(notches)

    def do(self, cmd: Command) -> Optional[str]:
        a = cmd.action
        if "autoscroll" not in a:                  # any other command stops "keep scrolling"
            self.auto = 0
        if a.get("builtin"):
            return a["builtin"]
        if self.dry:
            print(f"[DRY RUN voice] {cmd.phrase}: {json.dumps({k: v for k, v in a.items() if k != 'aliases'})}")
        if "autoscroll" in a:
            self.auto = int(a["autoscroll"])
            self._last_auto = 0.0
        if cmd.keys and not self.dry:
            for _ in range(max(1, int(a.get("repeat", 1)))):
                for combo in cmd.keys:
                    self.pg.hotkey(*combo, interval=0.0)
        if "text" in a and not self.dry:
            self.pg.write(str(a["text"]), interval=0.005)
        if "run" in a and not self.dry:
            try:
                args = str(a.get("args", "")).split()
                subprocess.Popen([str(a["run"])] + args)
            except Exception as e:  # noqa: BLE001
                print(f"[voice] could not run {a['run']}: {e}")
        if "scroll" in a:
            self._wheel(int(a["scroll"]))
        if "hscroll" in a:
            self._wheel(int(a["hscroll"]), horizontal=True)
        if "mouse" in a and not self.dry:
            kind = a["mouse"]
            if kind == "double":
                self.pg.doubleClick()
            else:
                self.pg.click(button=kind)
        return None

    def tick(self, now: float):
        if self.auto and now - self._last_auto >= self.AUTOSCROLL_EVERY:
            self._last_auto = now
            self._wheel(self.auto)


# --------------------------------------------------------------------------- #
# "talk to the hand": mute audio for a camera-facing palm with fingers spread
# --------------------------------------------------------------------------- #
def ensure_hand_model() -> Path:
    if HAND_MODEL_PATH.exists() and HAND_MODEL_PATH.stat().st_size > 1_000_000:
        return HAND_MODEL_PATH
    APP_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading MediaPipe hand_landmarker.task (~8 MB) ...")
    tmp = HAND_MODEL_PATH.with_suffix(".part")
    urllib.request.urlretrieve(HAND_MODEL_URL, tmp)
    tmp.replace(HAND_MODEL_PATH)
    return HAND_MODEL_PATH


def is_open_palm(landmarks, world_landmarks, handedness, frame_aspect) -> bool:
    """Require a visible, camera-facing palm and five extended, separated fingers."""
    if len(landmarks) != 21 or len(world_landmarks) != 21 or not handedness:
        return False
    side = handedness[0]
    if side.category_name not in ("Left", "Right") or not side.score >= 0.7:
        return False
    image = np.array([(p.x, p.y) for p in landmarks])
    points = np.array([(p.x, p.y, p.z) for p in world_landmarks])
    if not np.isfinite(points).all() or not np.isfinite(image).all():
        return False
    if (image < 0).any() or (image > 1).any():
        return False

    palm_width = np.linalg.norm(points[5] - points[17])
    normal = np.cross(points[5] - points[0], points[17] - points[0])
    normal_size = np.linalg.norm(normal)
    if palm_width < 1e-6 or normal_size < 1e-8:
        return False
    # MediaPipe's depth grows away from the camera. Its handedness and landmark
    # winding both reverse on a mirrored frame, so the facing test still holds.
    facing = normal[2] * (-1 if side.category_name == "Right" else 1) / normal_size
    if facing < 0.65:
        return False

    for base in (1, 5, 9, 13, 17):
        finger = points[base:base + 4]
        length = np.linalg.norm(np.diff(finger, axis=0), axis=1).sum()
        straightness = 0.85 if base == 1 else 0.9
        if length < 1e-6 or np.linalg.norm(finger[-1] - finger[0]) < straightness * length:
            return False
        # A straight finger folded toward the wrist is not an extended finger.
        if (np.linalg.norm(finger[-1] - points[0]) <
                np.linalg.norm(finger[-2] - points[0]) + 0.1 * palm_width):
            return False

    # Measure the visible finger gaps in the camera image, in equal units on both axes.
    image[:, 0] *= frame_aspect
    visible_palm_width = np.linalg.norm(image[5] - image[17])
    for a, b in ((4, 8), (8, 12), (12, 16), (16, 20)):
        base_a = 2 if a == 4 else a - 3
        base_b = b - 3
        gap = np.linalg.norm(image[a] - image[b])
        base_gap = np.linalg.norm(image[base_a] - image[base_b])
        if gap < max(0.3 * visible_palm_width, 1.25 * base_gap):
            return False
    return True


class HandGate:
    """Mute after `on_checks` open-palm checks; hold for `off_s` after the gesture ends."""

    def __init__(self, landmarker=None, every=3, on_checks=2, off_s=0.7, width=320):
        self.every, self.on_checks, self.off_s, self.width = max(1, every), on_checks, off_s, width
        self.lm = landmarker if landmarker is not None else self._make_landmarker()
        self.n = 0
        self.hits = 0
        self.up = False
        self.last_seen = -1e9
        self._ts = 0

    @staticmethod
    def _make_landmarker():
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_hand_model())),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        return vision.HandLandmarker.create_from_options(opts)

    def _detect(self, frame_bgr) -> bool:
        import cv2
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (self.width, max(1, int(h * self.width / w))))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self._ts += 33 * self.every
        res = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._ts)
        return any(is_open_palm(image, world, side, small.shape[1] / small.shape[0])
                   for image, world, side in
                   zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness))

    def update(self, frame_bgr, t: float, seen: Optional[bool] = None) -> bool:
        self.n += 1
        if seen is not None or self.n % self.every == 0:
            if seen is None:
                seen = self._detect(frame_bgr)
            if seen:
                self.hits += 1
                self.last_seen = t
                if self.hits >= self.on_checks:
                    self.up = True
            else:
                self.hits = 0
        if self.up and t - self.last_seen > self.off_s:
            self.up = False
        return self.up


# --------------------------------------------------------------------------- #
# the listening thread
# --------------------------------------------------------------------------- #
def ensure_vosk_model(path_hint: Optional[str] = None) -> Path:
    if path_hint:
        return Path(path_hint)
    root = APP_DIR / "vosk"
    model = root / VOSK_MODEL_NAME
    if (model / "am").exists():
        return model
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Vosk speech model {VOSK_MODEL_NAME} (~40 MB) ...")
    zpath = root / (VOSK_MODEL_NAME + ".zip")
    urllib.request.urlretrieve(VOSK_MODEL_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    zpath.unlink(missing_ok=True)
    return model


class VoiceEngine(threading.Thread):
    """Reads the microphone, decides what was said, and queues ("do", Resolved) for the main loop."""

    def __init__(self, router: Router, out: "queue.Queue", mic=None, min_conf=0.55,
                 model_path=None, verbose=False, whisper_model="base.en", dictation_gap=0.8):
        super().__init__(daemon=True, name="voice")
        self.router, self.out, self.mic = router, out, mic
        self.min_conf, self.model_path, self.verbose = min_conf, model_path, verbose
        self.muted = False            # set from the camera loop (open palm)
        self.paused = False           # "pause listener"
        self.status = "STARTING"
        self.error: Optional[str] = None
        self.heard = ("", 0.0)
        self._halt = threading.Event()
        self._dirty = True
        self._pre: list[bytes] = []
        self._utt: list[bytes] = []
        self._in_speech = False
        self._was_muted = False
        self.whisper_model, self.dictation_gap = whisper_model, dictation_gap
        self.whisper = None
        self._dictation_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dictation")
        self._dictation_buffer = None
        self._dictation_future = None
        self._dictation_reported = False
        self._dictation_target = None
        self._dictation_generation = 0
        self._dictation_lock = threading.Lock()
        self._was_paused = False

    # --- public ---------------------------------------------------------- #
    def reload(self):
        self._dirty = True

    def stop(self):
        self._halt.set()
        self.cancel_dictation()
        self._dictation_pool.shutdown(wait=False, cancel_futures=True)

    def cancel_dictation(self):
        with self._dictation_lock:
            if self._dictation_buffer is not None or self._dictation_future is not None:
                print("[dictation] cancelled; returning to commands", flush=True)
            self._dictation_generation += 1
            self._dictation_buffer = None
            future, self._dictation_future = self._dictation_future, None
            if future is not None:
                future.cancel()
        self._dirty = True

    def begin_dictation(self):
        if self.whisper is None:
            print("[dictation] Whisper unavailable; dictation was not started", flush=True)
            return
        with self._dictation_lock:
            if self.paused or self.muted or self._halt.is_set():
                return
            self._dictation_generation += 1
            self._dictation_target = foreground_window()
            self._dictation_buffer = SpeechBuffer(gap=self.dictation_gap)
            self._dictation_reported = False
        self.status = "DICTATING"
        self._reset()
        print(f"[dictation] ready: speak, then pause {self.dictation_gap:g}s to finish", flush=True)

    def finish_dictation(self):
        with self._dictation_lock:
            self._dictation_future = None
        self._dirty = True
        self.status = "LISTENING"

    def _record_dictation(self, data):
        with self._dictation_lock:
            buffer = self._dictation_buffer
            generation = self._dictation_generation
        if buffer is None:
            return
        audio = buffer.feed(data)
        if audio is None:
            return
        with self._dictation_lock:
            if generation != self._dictation_generation or self._halt.is_set():
                return
            self._dictation_buffer = None
            self._reset()
            if not audio:
                self.status = "LISTENING"
                print("[dictation] no speech captured; returning to commands", flush=True)
                return
            self.status = "TRANSCRIBING"
            print(f"[dictation] transcribing {len(audio) / (SAMPLE_RATE * 2):.1f}s locally", flush=True)
            self._dictation_future = self._dictation_pool.submit(self.whisper.transcribe, audio)

    def _poll_dictation(self):
        with self._dictation_lock:
            future = self._dictation_future
            if future is None or not future.done() or self._dictation_reported:
                return
            self._dictation_reported = True
            try:
                text = future.result()
            except Exception as e:  # noqa: BLE001
                print(f"[dictation] transcription failed: {e}", flush=True)
                self._dictation_future = None
                self._dirty = True
                return
            print(f"[dictation] transcript: {text!r}", flush=True)
            self.heard = (text or "<no words recognized>", time.time())
            self.out.put(("dictation", (text, self._dictation_target, self._dictation_generation)))

    # --- internals -------------------------------------------------------- #
    def _grammar(self, only_control=False) -> str:
        if only_control:
            phrases = [p for p, c in self.router.table.items() if c.builtin in ("resume_listener", "stop_all")]
        else:
            phrases = list(self.router.table) + CONTROL_PHRASES
        return json.dumps(sorted(set(phrases)) + ["[unk]"])

    def _build(self):
        KR = self._KR
        self.rec_cmd = KR(self.model, SAMPLE_RATE, self._grammar())
        self.rec_paused = KR(self.model, SAMPLE_RATE, self._grammar(only_control=True))
        for r in (self.rec_cmd, self.rec_paused):
            r.SetWords(True)
        self.rec_free = KR(self.model, SAMPLE_RATE)
        self._dirty = False
        self._reset()

    def _reset(self):
        for r in (self.rec_cmd, self.rec_paused):
            try:
                r.Reset()
            except Exception:  # noqa: BLE001
                self._dirty = True
        self._pre, self._utt, self._in_speech = [], [], False

    def run(self):
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except Exception as e:  # noqa: BLE001
            self.error = f"missing package ({e}). pip install vosk sounddevice"
            self.status = "OFF"
            print(f"[voice] {self.error}")
            return
        try:
            print(f"[dictation] loading local Whisper {self.whisper_model} ...", flush=True)
            self.whisper = LocalWhisper(self.whisper_model)
            print("[dictation] ready; say 'dictate' before speaking your text", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[dictation] unavailable: {e}", flush=True)
        try:
            SetLogLevel(-1)
            self._KR = KaldiRecognizer
            self.model = Model(str(ensure_vosk_model(self.model_path)))
            self._build()
            q: queue.Queue = queue.Queue(maxsize=200)

            def callback(indata, frames, t, status):  # noqa: ARG001
                try:
                    q.put_nowait(bytes(indata))
                except queue.Full:
                    pass

            stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, device=self.mic,
                                       dtype="int16", channels=1, callback=callback)
            device = sd.query_devices(stream.device, "input")
            print(f"[voice] microphone: {device['name']} ({SAMPLE_RATE} Hz)", flush=True)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            self.status = "OFF"
            print(f"[voice] could not start: {e}")
            return
        with stream:
            print(f"[voice] listening ({len(self.router.table)} phrases). "
                  f"Say 'pause listener' / 'stop listener'.")
            while not self._halt.is_set():
                try:
                    data = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    self._step(data)
                except Exception as e:  # noqa: BLE001
                    print(f"[voice] error: {e}")
        self.status = "OFF"

    def _step(self, data: bytes):
        if self._dirty:
            self._build()
        muted = self.muted
        if muted != self._was_muted:
            self._was_muted = muted
            if muted:
                self.cancel_dictation()
            self._reset()
            print(f"[voice] {'MUTED (open palm)' if muted else 'palm gesture ended; listening resumed'}",
                  flush=True)
        if muted:
            self.status = "MUTED (open palm)"
            return
        if self.paused != self._was_paused:
            self._was_paused = self.paused
            if self.paused:
                self.cancel_dictation()
            self._reset()
        if self._dictation_buffer is not None:
            self.status = "DICTATING"
            # Keep the explicit stop phrases available without routing dictated commands.
            if self.rec_paused.AcceptWaveform(data):
                heard = json.loads(self.rec_paused.Result())
                if norm_text(heard.get("text", "")) in ("stop pilot", "stop listener"):
                    resolved = self.router.table.get(norm_text(heard["text"]))
                    words = heard.get("result") or []
                    if resolved is not None and resolved.builtin == "stop_all" and words and all(
                            word.get("conf", 0) >= self.min_conf for word in words):
                        self.cancel_dictation()
                        self._emit(Resolved(resolved.phrase, resolved, "exact"), heard["text"])
                        return
            self._record_dictation(data)
            return
        self._poll_dictation()
        busy = self._dictation_future is not None
        self.status = "PAUSED" if self.paused else "TRANSCRIBING" if busy else "LISTENING"
        rec = self.rec_paused if self.paused or busy else self.rec_cmd
        (self._utt if self._in_speech else self._pre).append(data)
        if not self._in_speech and len(self._pre) > 4:      # keep ~0.5 s of lead-in only
            self._pre.pop(0)
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            audio = b"".join(self._utt if self._in_speech else self._pre)
            self._pre, self._utt, self._in_speech = [], [], False
            self._on_utterance(res, audio, rec is self.rec_paused)
        elif not self._in_speech and json.loads(rec.PartialResult()).get("partial"):
            self._in_speech, self._utt, self._pre = True, list(self._pre), []

    def _transcribe(self, audio: bytes) -> str:
        if len(audio) < 9600:                               # < 0.3 s
            return ""
        try:
            self.rec_free.Reset()
        except Exception:  # noqa: BLE001
            self.rec_free = self._KR(self.model, SAMPLE_RATE)
        parts = []
        if self.rec_free.AcceptWaveform(audio + b"\x00" * 6400):
            parts.append(json.loads(self.rec_free.Result()).get("text", ""))
        parts.append(json.loads(self.rec_free.FinalResult()).get("text", ""))
        return " ".join(p for p in parts if p)

    def _on_utterance(self, res: dict, audio: bytes, paused_rec: bool):
        raw = res.get("text", "").strip()
        words = res.get("result") or []
        conf = sum(w.get("conf", 1.0) for w in words) / len(words) if words else 1.0
        confidence = f"{conf:.0%}" if words else "not reported"
        print(f'[voice] command guess: {raw!r} (confidence: {confidence})', flush=True)
        known = "[unk]" not in raw
        cmd = self.router.table.get(norm_text(raw)) if known else None
        combo = parse_combo(raw) if known else None
        if paused_rec:                                      # paused: only resume / stop are honoured
            if cmd is not None and cmd.builtin in ("resume_listener", "stop_all") and conf >= self.min_conf:
                self._emit(Resolved(cmd.phrase, cmd, "exact"), raw)
            else:
                print("[voice] listener paused; ignoring speech", flush=True)
            return
        text = ""
        if self.verbose or (cmd is None and combo is None) or conf < self.min_conf:
            text = self._transcribe(audio)
            print(f'[voice] transcript: {text!r}' if text else
                  "[voice] transcript: <no words recognized>", flush=True)
        if cmd is not None and conf >= self.min_conf:
            self._emit(Resolved(cmd.phrase, cmd, "exact"), raw)
            return
        if combo is not None and conf >= self.min_conf:
            self._emit(self.router.route(raw), raw)
            return
        r = self.router.route(text) if text else None
        if r is not None and r.command.builtin in ("resume_listener",):
            r = None                                        # not paused, nothing to resume
        self._emit(r, text or raw or "<no words recognized>")

    def _emit(self, r: Optional[Resolved], text: str):
        note = f'"{text}" -> {r.label} ({r.how}{"" if r.how != "fuzzy" else f" {r.score:.2f}"})' \
            if r else f'"{text}" -> no match'
        self.heard = (note, time.time())
        print(f"[voice] {note}", flush=True)
        if r is not None:
            if r.command.builtin == "dictate":
                self.begin_dictation()
            else:
                self.out.put(("do", r))


# --------------------------------------------------------------------------- #
# what pilot.py talks to
# --------------------------------------------------------------------------- #
class VoiceSystem:
    def __init__(self, args, mouse, script_dir: Path):
        self.cfg_path = Path(args.commands) if args.commands else script_dir / "commands.json"
        table, settings, problems = load_commands(self.cfg_path)
        for p in problems:
            print(f"[voice] skipped command {p}")
        embedder = None
        try:
            embedder = make_embedder(args.embed_backend, args.embed_model)
        except Exception as e:  # noqa: BLE001
            print(f"[voice] embedding model unavailable: {e}")
        self.router = Router(table, embedder, args.embed_threshold, args.embed_margin,
                             verbose=args.voice_verbose)
        self.events: queue.Queue = queue.Queue()
        min_conf = args.voice_confidence if args.voice_confidence is not None \
            else float(settings.get("minConfidence", 0.55))
        self.engine = VoiceEngine(self.router, self.events, args.mic, min_conf,
                                  args.vosk_model, args.voice_verbose, args.whisper_model, args.dictation_gap)
        self.act = Actuator(mouse.pg, dry=args.dry_run)
        self.hand: Optional[HandGate] = None
        if not args.no_hand_mute:
            try:
                self.hand = HandGate(every=args.hand_every)
            except Exception as e:  # noqa: BLE001
                print(f"[voice] hand muting off: {e}")
        self._mtime = self._stat()
        self._next_poll = 0.0
        print(f"[voice] {len(table)} phrases from {self.cfg_path.name}; fuzzy match: "
              f"{embedder if embedder else 'off'}; hand-mute: {'on' if self.hand else 'off'}")

    def _stat(self):
        try:
            return self.cfg_path.stat().st_mtime
        except OSError:
            return None

    def start(self):
        self.engine.start()

    def stop(self):
        self.engine.stop()

    def clear_events(self):
        self.engine.cancel_dictation()
        while not self.events.empty():
            try:
                self.events.get_nowait()
            except queue.Empty:
                break

    def _poll_config(self, now: float):
        if now < self._next_poll:
            return
        self._next_poll = now + 1.0
        m = self._stat()
        if m is None or m == self._mtime:
            return
        self._mtime = m
        try:
            table, _, problems = load_commands(self.cfg_path)
        except Exception as e:  # noqa: BLE001
            print(f"[voice] commands.json not reloaded: {e}")
            return
        for p in problems:
            print(f"[voice] skipped command {p}")
        self.router.set_commands(table)
        self.engine.reload()
        print(f"[voice] reloaded {len(table)} phrases")

    def pump(self, frame, now: float) -> list[str]:
        """Call once per camera frame. Returns pilot-level builtins for pilot.py to handle."""
        if self.hand is not None:
            try:
                was_muted = self.engine.muted
                self.engine.muted = self.hand.update(frame, now)
                if self.engine.muted and not was_muted:
                    self.engine.cancel_dictation()
            except Exception as e:  # noqa: BLE001
                print(f"[voice] hand muting stopped: {e}")
                self.hand, self.engine.muted = None, False
        self._poll_config(now)
        out: list[str] = []
        while True:
            try:
                kind, r = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "dictation":
                text, target, generation = r
                try:
                    if (generation != self.engine._dictation_generation or self.engine.muted
                            or self.engine.paused or self.engine._halt.is_set()):
                        print("[dictation] cancelled text was not inserted", flush=True)
                    elif text and not self.act.dry and foreground_window() != target:
                        print("[dictation] window changed; text printed above was not inserted", flush=True)
                    elif text:
                        if self.act.dry:
                            print(f"[DRY RUN dictation] would type: {text}", flush=True)
                        else:
                            type_text(text)
                            print("[dictation] text inserted; returning to commands", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[dictation] could not insert text: {e}", flush=True)
                finally:
                    if generation == self.engine._dictation_generation:
                        self.engine.finish_dictation()
                continue
            b = self.act.do(r.command)
            if b == "stop_all":
                self.clear_events()
                return [b]
            if b == "pause_listener":
                self.engine.paused = True
                self.engine.cancel_dictation()
            elif b == "resume_listener":
                self.engine.paused = False
            elif b:
                out.append(b)
        self.act.tick(now)
        return out

    def hud_line(self) -> str:
        return f"VOICE: {self.engine.status}" + (f" ({self.engine.error[:40]})" if self.engine.error else "")

    def hud_heard(self) -> str:
        text, t = self.engine.heard
        return text if text and time.time() - t < 3.0 else ""
