#!/usr/bin/env python3
"""
pilot.py - hands-free cursor control with frame-counted wink clicking
========================================================================

* Uses calm, jitter-free bubble-glide cursor movement.
* LEFT wink = primary click (RIGHT wink = right click, optional)
* Hold the primary wink: single clicks at frames 7, 14, and 21, then reopen to reset.
* --right flag swaps roles: your RIGHT eye becomes primary click, left eye = right click
* Secondary wink: one right click per deliberate hold.
* Webcam is assumed to sit top-middle of the screen, so "looking straight
  into the camera" is calibrated as the top-middle of the screen.
* Guided calibration at start (9 screen points + a wink calibration).
  It is saved to ~/.eye_mouse/profile.json and re-used next time.

Install (run this with *Windows* Python, not inside WSL):
    pip install mediapipe opencv-python numpy pyautogui pynput

Run:
    python pilot.py --dry-run        # test tracking WITHOUT moving your real mouse
    python pilot.py                  # the real thing
    python pilot.py --right          # right-eye wink = left click (left-eye = right click)
    python pilot.py --input auto     # let calibration pick head / eyes / hybrid
    python pilot.py --control joystick
    python pilot.py --recalibrate

Safety / controls
    Close BOTH eyes for ~1.2 s ........ pause / resume (works hands-free)
    F8 ................................ pause / resume   (global hotkey)
    F7 ................................ re-centre        (global hotkey)
    F9 ................................ recalibrate      (global hotkey)
    F10 ............................... quit             (global hotkey)
    C .................................. re-centre        (global hotkey)
    R .................................. recalibrate      (global hotkey)
    Preview window keys: q/Esc quit, p pause, r recalibrate, w wink-calibrate,
                         c re-centre, m cycle cursor mode, i cycle input
    Ctrl+C in the terminal also quits.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
APP_DIR = Path.home() / ".eye_mouse"
MODEL_PATH = APP_DIR / "face_landmarker.task"
PROFILE_PATH = APP_DIR / "profile.json"

# MediaPipe 478-point mesh. Eye corners are ordered (image-left, image-right)
# in an un-mirrored frame so both eyes share the same horizontal convention.
EYE_R = (33, 133)       # subject's RIGHT eye (appears on the image left)
EYE_L = (362, 263)      # subject's LEFT  eye (appears on the image right)
IRIS_CENTRES = (468, 473)
PREVIEW_POINTS = (33, 133, 362, 263, 468, 473, 1)

# feature vector = [yaw deg, pitch deg, gaze_x, gaze_y]
FEATURE_FLOOR = np.array([0.5, 0.5, 0.004, 0.004])   # min std used when normalising
INPUT_SETS = {"head": [0, 1], "eyes": [2, 3], "hybrid": [0, 1, 2, 3]}
INPUT_ORDER = ["head", "eyes", "hybrid"]
CONTROL_ORDER = ["absolute", "joystick"]

# 3x3 calibration grid as screen fractions. The webcam is top-middle, so
# "look into the camera" is (0.5, 0.0).
CAL_POINTS = [
    (0.5, 0.0, "Look straight into the CAMERA (top-middle)"),
    (0.5, 0.5, "Look at the CENTRE of the screen"),
    (0.5, 1.0, "Look at the BOTTOM-middle"),
    (0.0, 0.5, "Look at the LEFT-middle"),
    (1.0, 0.5, "Look at the RIGHT-middle"),
    (0.0, 0.0, "Look at the TOP-LEFT corner"),
    (1.0, 0.0, "Look at the TOP-RIGHT corner"),
    (0.0, 1.0, "Look at the BOTTOM-LEFT corner"),
    (1.0, 1.0, "Look at the BOTTOM-RIGHT corner"),
]
GAZE_EMA = 0.35     # per-frame weight of the newest iris sample (lower = smoother)
SETTLE_S = 1.0      # time to move head/eyes to the dot before sampling
CAPTURE_S = 1.2     # sampling time per dot

DEFAULT_WINK = {
    "swap": False,   # True if eyeBlinkLeft turned out to be the user's right eye
    "left": {"enabled": True, "close": 0.50, "other_max": 0.40},
    "right": {"enabled": True, "close": 0.50, "other_max": 0.40},
}


class CalibrationAborted(Exception):
    pass


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def put_text(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def put_centered(img, text, cy, scale, color=(255, 255, 255), thick=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    put_text(img, text, ((img.shape[1] - tw) // 2, cy), scale, color, thick)


def beep(high=True):
    try:
        import winsound
        winsound.Beep(1000 if high else 600, 80)
    except Exception:
        pass


def check_platform():
    if sys.platform.startswith("linux") and "microsoft" in platform.uname().release.lower():
        sys.exit("This looks like WSL. WSL cannot reach your webcam or move the Windows cursor.\n"
                 "Run this script with the Windows Python instead, e.g. in PowerShell:\n"
                 "    py -3.12 pilot.py")


def ensure_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    script_dir = Path(__file__).resolve().parent
    for cand in (Path("face_landmarker.task"), script_dir / "face_landmarker.task"):
        if cand.exists() and cand.stat().st_size > 1_000_000:
            return cand
    APP_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading MediaPipe face_landmarker.task (~4 MB) ...")
    tmp = MODEL_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not download the model ({e}).\nDownload it manually from\n  {MODEL_URL}\n"
                 f"and save it as\n  {MODEL_PATH}")
    return MODEL_PATH


def open_camera(index: int, width: int, height: int):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {index}. Try --camera 1, and close other apps using the webcam.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_frame(cam):
    ok, frame = cam.read()
    if not ok or frame is None:
        time.sleep(0.01)
        return None
    return frame


# --------------------------------------------------------------------------- #
# face tracking (MediaPipe Face Landmarker, Tasks API)
# --------------------------------------------------------------------------- #
@dataclass
class Obs:
    t: float
    yaw: float            # degrees, sign is learned during calibration
    pitch: float
    gaze: np.ndarray      # (2,) iris offset in eye-widths (last valid value)
    gaze_fresh: bool
    blink_l: float        # USER's left eye, 0 = open .. 1 = closed
    blink_r: float
    jaw: float
    pts: np.ndarray       # (478, 2) normalised image coords

    def features(self) -> np.ndarray:
        return np.array([self.yaw, self.pitch, self.gaze[0], self.gaze[1]], float)


class FaceTracker:
    def __init__(self, model_path, swap_eyes: bool = False):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.lm = vision.FaceLandmarker.create_from_options(options)
        self.swap_eyes = swap_eyes
        self._last_ts = -1
        self.last_gaze = np.zeros(2)

    @staticmethod
    def head_angles(M) -> tuple[float, float]:
        """yaw/pitch of the face's forward axis (3rd column of the rotation)."""
        R = np.asarray(M, float)[:3, :3]
        f = R[:, 2]
        f = f / (np.linalg.norm(f) + 1e-9)
        yaw = math.degrees(math.atan2(f[0], abs(f[2])))
        pitch = math.degrees(math.atan2(f[1], math.hypot(f[0], f[2])))
        return yaw, pitch

    @staticmethod
    def eye_gaze(pix, a, b) -> Optional[np.ndarray]:
        """Iris offset from the eye-corner midpoint, in units of eye width,
        expressed along (corner->corner, perpendicular) so it survives head roll."""
        A, B = pix[a], pix[b]
        c = (A + B) / 2
        u = B - A
        w = float(np.linalg.norm(u))
        if w < 1.0:
            return None
        u = u / w
        v = np.array([-u[1], u[0]])
        cand = pix[list(IRIS_CENTRES)] - c
        d = cand[int(np.argmin(np.linalg.norm(cand, axis=1)))]   # iris nearest to this eye
        return np.array([d @ u, d @ v]) / w

    def process(self, frame_bgr, t: float) -> Optional[Obs]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = int(t * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        res = self.lm.detect_for_video(mp_img, ts)
        if not res.face_landmarks or not res.facial_transformation_matrixes:
            return None

        pts = np.array([[p.x, p.y] for p in res.face_landmarks[0]], np.float32)
        pix = pts * np.array([w, h], np.float32)
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        a, b = bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)     # subject's L / R
        bl, br = (b, a) if self.swap_eyes else (a, b)

        yaw, pitch = self.head_angles(res.facial_transformation_matrixes[0])

        g_r = self.eye_gaze(pix, *EYE_R)
        g_l = self.eye_gaze(pix, *EYE_L)
        w_r = clamp((0.6 - b) / 0.3, 0.0, 1.0)          # ignore a closing eye
        w_l = clamp((0.6 - a) / 0.3, 0.0, 1.0)
        fresh = g_r is not None and g_l is not None and (w_r + w_l) > 0
        if fresh:
            self.last_gaze = (w_r * g_r + w_l * g_l) / (w_r + w_l)
        return Obs(t, yaw, pitch, self.last_gaze.copy(), fresh, bl, br,
                   bs.get("jawOpen", 0.0), pts)


# --------------------------------------------------------------------------- #
# filtering and mapping
# --------------------------------------------------------------------------- #
class OneEuro:
    """1-Euro filter (Casiez et al. 2012): heavy smoothing when slow, light when fast."""

    def __init__(self, min_cutoff: float, beta: float = 0.012, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self, value=None, t=None):
        self.x = None if value is None else np.asarray(value, float).copy()
        self.t = t if value is not None else None
        self.dx = np.zeros(2)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None or self.t is None:
            self.x, self.t, self.dx = x.copy(), t, np.zeros_like(x)
            return x.copy()
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class Mapper:
    """Ridge regression: feature vector -> screen pixels. Because it is *learned*
    from the calibration dots, axis directions/signs never have to be guessed."""

    def __init__(self, lam: float = 0.5):
        self.lam = lam

    def fit(self, F, P, cols):
        self.cols = list(cols)
        X = np.asarray(F, float)[:, self.cols]
        self.mu = X.mean(0)
        self.sd = np.maximum(X.std(0), FEATURE_FLOOR[self.cols])
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        reg = self.lam * np.eye(Z.shape[1])
        reg[-1, -1] = 0.0                       # don't shrink the bias
        self.W = np.linalg.solve(Z.T @ Z + reg, Z.T @ np.asarray(P, float))
        return self

    def predict(self, f):
        z = (np.asarray(f, float)[self.cols] - self.mu) / self.sd
        return np.append(z, 1.0) @ self.W


def loo_error(F, P, cols, W, H):
    """Leave-one-out RMS error as % of screen width / height."""
    F, P = np.asarray(F, float), np.asarray(P, float)
    err = []
    for i in range(len(F)):
        keep = np.arange(len(F)) != i
        err.append(Mapper().fit(F[keep], P[keep], cols).predict(F[i]) - P[i])
    e = np.array(err)
    return (float(np.sqrt((e[:, 0] ** 2).mean()) / W * 100),
            float(np.sqrt((e[:, 1] ** 2).mean()) / H * 100))


def build_mapper(F, P, W, H, input_name, verbose=True):
    table = {name: loo_error(F, P, cols, W, H) for name, cols in INPUT_SETS.items()}
    if verbose:
        print("\nCalibration quality (leave-one-out error, % of screen; lower is better):")
        for name, (ex, ey) in table.items():
            print(f"  {name:7s} x \u00b1{ex:4.1f}%   y \u00b1{ey:4.1f}%")
    if input_name == "auto":
        input_name = min(table, key=lambda k: math.hypot(*table[k]))
    if verbose:
        print(f"Using input: {input_name}\n")
    return Mapper().fit(F, P, INPUT_SETS[input_name]), input_name


# --------------------------------------------------------------------------- #
# cursor engine (dynamic bubble smoothing)
# --------------------------------------------------------------------------- #
class CursorEngine:
    def __init__(self, W, H, mapper, control="absolute", gain=1.0, smoothing=0.6,
                 speed=1.0, deadzone=0.12):
        self.W, self.H = W, H
        self.mapper, self.control = mapper, control
        self.gain, self.speed, self.dz = gain, speed, deadzone
        self.filt = OneEuro(min_cutoff=3.0 * (0.05 ** clamp(smoothing, 0.0, 1.0)))
        self.centre = np.array([(W - 1) / 2, (H - 1) / 2])
        self.half = np.array([W / 2, H / 2])
        self.offset = np.zeros(2)
        self.pos = self.centre.copy()
        self.hist: deque = deque(maxlen=90)
        self.last_t: Optional[float] = None
        self._feats: Optional[np.ndarray] = None   # features with gaze low-passed

        # Two-stage dynamic bubble smoothing from version 2/3 (calm, jitter-free glide)
        self.bubble = self.centre.copy()
        self.bubble_radius = 40.0
        self.min_bubble_radius = 14.0
        self.max_bubble_radius = 65.0
        self.radius_alpha = 0.20
        self.cursor_alpha = 0.28

    def _smooth_feats(self, feats) -> np.ndarray:
        """Iris position is ~3x noisier than head angle, so low-pass just the gaze part."""
        f = np.asarray(feats, float).copy()
        if self._feats is not None:
            f[2:] = GAZE_EMA * f[2:] + (1 - GAZE_EMA) * self._feats[2:]
        self._feats = f.copy()
        return f

    def predict(self, feats) -> np.ndarray:
        p = self.mapper.predict(feats) + self.offset
        return self.centre + self.gain * (p - self.centre)

    def update(self, t, feats) -> np.ndarray:
        dt = 0.0 if self.last_t is None else min(t - self.last_t, 0.1)
        self.last_t = t
        q = self.filt(self.predict(self._smooth_feats(feats)), t)
        if self.control == "absolute":
            # Target from 1-Euro filtered prediction
            target = np.clip(q, 0, [self.W - 1, self.H - 1])

            # Stage 1: Bubble smoothing with velocity-adaptive responsiveness
            speed = float(np.hypot(target[0] - self.bubble[0], target[1] - self.bubble[1]))
            bubble_alpha = float(np.interp(speed, [0, 200], [0.03, 0.25]))
            self.bubble = (bubble_alpha * target) + ((1.0 - bubble_alpha) * self.bubble)

            # Stage 2: Dynamic bubble radius (expands on rapid movement, shrinks at rest)
            target_radius = float(np.interp(speed, [0, 180], [self.min_bubble_radius, self.max_bubble_radius]))
            self.bubble_radius = (self.radius_alpha * target_radius) + ((1.0 - self.radius_alpha) * self.bubble_radius)

            # Stage 3: Cursor glides towards bubble center; deadband calm when inside bubble
            dist = float(np.hypot(self.bubble[0] - self.pos[0], self.bubble[1] - self.pos[1]))
            active_alpha = self.cursor_alpha if dist > self.bubble_radius else self.cursor_alpha * 0.35
            self.pos = (active_alpha * self.bubble) + ((1.0 - active_alpha) * self.pos)

            # Screen boundary clamp
            self.pos = np.clip(self.pos, 0, [self.W - 1, self.H - 1])
        else:
            s = np.clip((q - self.centre) / self.half, -1.5, 1.5)
            mag = np.maximum(np.abs(s) - self.dz, 0.0) / (1.0 - self.dz)
            vel = np.sign(s) * mag ** 1.5 * self.speed * np.array([self.W, self.H])
            self.pos = np.clip(self.pos + vel * dt, 0, [self.W - 1, self.H - 1])
        self.hist.append((t, self.pos.copy()))
        return self.pos

    def recenter(self, feats):
        """Treat the current pose as 'looking at the screen centre'."""
        f = self._feats if self._feats is not None else feats
        self.offset = self.centre - self.mapper.predict(f)
        self.filt.reset()
        self.bubble = self.centre.copy()
        self.pos = self.centre.copy()

    def rewind(self, t_back):
        """Jump back to where the cursor was just before a blink/wink started."""
        for ts, p in self.hist:
            if ts >= t_back:
                self.pos = p.copy()
                self.bubble = p.copy()
                return

    def release(self, t):
        """Resume after a freeze without a jump."""
        if self.control == "absolute":
            self.filt.reset(self.pos, t)
            self.bubble = self.pos.copy()
        else:
            self.filt.reset()


# --------------------------------------------------------------------------- #
# gesture detectors
# --------------------------------------------------------------------------- #
class WinkDetector:
    """Primary wink clicks at frames 7/14/21; secondary wink clicks once."""

    def __init__(self, prof, long_hold=0.18, min_frames=2,
                 freeze_thr=0.30, settle=0.10, pause_hold=1.2):
        self.p = prof
        self.long_hold = long_hold
        self.min_frames = min_frames
        self.freeze_thr, self.settle, self.pause_hold = freeze_thr, settle, pause_hold
        self.clear()
        self.last_closed = -1e9
        self.both_since: Optional[float] = None
        self.pause_armed = True

    def clear(self):
        self.state = {s: {"since": None, "frames": 0, "fired_long": False, "peak": 0.0}
                      for s in ("left", "right")}

    def update(self, t, bl, br) -> list[tuple[str, bool]]:
        events = []
        for side, (me, other) in {"left": (bl, br), "right": (br, bl)}.items():
            cfg, st = self.p[side], self.state[side]
            cond = (cfg["enabled"] and me >= cfg["close"] and other <= cfg["other_max"]
                    and (me - other) >= 0.20)
            if cond:
                if st["since"] is None:
                    st["since"], st["frames"] = t, 0
                st["frames"] += 1
                st["peak"] = max(st["peak"], me)
                dur = t - st["since"]
                if side == "left":
                    if st["frames"] == 7:
                        events.append((side, True))
                    elif st["frames"] == 14:
                        events.append((side, True))
                    elif st["frames"] == 21:
                        events.append((side, True))
                elif not st["fired_long"] and dur >= self.long_hold and st["frames"] >= self.min_frames:
                    st["fired_long"] = True
                    events.append((side, True))

            elif st["since"] is not None:
                # Re-arm as soon as eye relaxes
                reopened = me < cfg["close"] * 0.82 or me < st["peak"] - 0.15 or (me - other) < 0.18
                if reopened:
                    st.update(since=None, frames=0, fired_long=False, peak=0.0)

        if max(bl, br) >= self.freeze_thr:
            self.last_closed = t

        if min(bl, br) >= 0.6:                          # both eyes held shut -> pause
            if self.both_since is None:
                self.both_since = t
            if self.pause_armed and t - self.both_since >= self.pause_hold:
                self.pause_armed = False
                events.append(("toggle_pause", False))
        else:
            self.both_since = None
            if max(bl, br) < 0.3:
                self.pause_armed = True
        return events

    def frozen(self, t) -> bool:
        return (t - self.last_closed) < self.settle


class MouthDrag:
    """Optional: hold mouth open ~0.3 s to toggle mouse-button-down (drag lock)."""

    def __init__(self, on=0.6, off=0.25, hold=0.30):
        self.on, self.off, self.hold = on, off, hold
        self.since: Optional[float] = None
        self.armed = True

    def update(self, t, jaw) -> bool:
        if jaw >= self.on:
            if self.since is None:
                self.since = t
            if self.armed and t - self.since >= self.hold:
                self.armed = False
                return True
        elif jaw <= self.off:
            self.since, self.armed = None, True
        return False


def analyse_wink(rest, left, right) -> dict:
    """rest/left/right: arrays (n, 2) of raw (eyeBlinkLeft, eyeBlinkRight)."""
    rest, left, right = (np.asarray(x, float) for x in (rest, left, right))

    def strongest(arr):
        d = np.abs(arr[:, 0] - arr[:, 1])
        idx = np.argsort(-d)[:max(5, len(arr) // 3)]
        return np.median(arr[idx], axis=0)

    open_ = np.median(rest, axis=0)
    wl, wr = strongest(left), strongest(right)
    score = (wl[0] - wl[1]) + (wr[1] - wr[0])           # >0: eyeBlinkLeft really is the user's left
    swap = bool(score < 0)
    li, ri = (1, 0) if swap else (0, 1)                 # indices of user's left / right eye

    out = {"swap": swap, "ok": abs(score) >= 0.3}
    for side, w, me, ot in (("left", wl, li, ri), ("right", wr, ri, li)):
        closed, other, op = float(w[me]), float(w[ot]), float(open_[me])
        out[side] = {
            "enabled": bool(closed - other >= 0.30),
            "close": float(clamp(op + 0.55 * (closed - op), 0.25, 0.70)),
            "other_max": float(clamp(other + 0.50 * (closed - other), 0.20, 0.60)),
        }
    return out


# --------------------------------------------------------------------------- #
# calibration UI (full-screen OpenCV window)
# --------------------------------------------------------------------------- #
class Fullscreen:
    NAME = "Eye Mouse - calibration"

    def __init__(self, W, H):
        self.W, self.H = W, H
        cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def blank(self):
        return np.full((self.H, self.W, 3), 18, np.uint8)

    def show(self, img):
        cv2.imshow(self.NAME, img)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            raise CalibrationAborted()

    def close(self):
        try:
            cv2.destroyWindow(self.NAME)
            cv2.waitKey(1)
        except cv2.error:
            pass


def draw_point_screen(img, mx, my, title, subtitle, idx, total, el, face_ok):
    H, W = img.shape[:2]
    s = H / 1080.0
    ty = int(0.74 * H) if my < H * 0.45 else int(0.26 * H)
    put_centered(img, f"[{idx + 1}/{total}]  {title}", ty, 1.2 * s, (255, 255, 255), max(1, int(2 * s)))
    put_centered(img, subtitle, ty + int(46 * s), 0.7 * s, (190, 190, 190), 1)
    if el < SETTLE_S:
        ring = int((22 + 60 * (1 - el / SETTLE_S)) * s)
        cv2.circle(img, (mx, my), ring, (120, 120, 120), max(1, int(3 * s)), cv2.LINE_AA)
    else:
        frac = clamp((el - SETTLE_S) / CAPTURE_S, 0, 1)
        cv2.ellipse(img, (mx, my), (int(36 * s),) * 2, -90, 0, 360 * frac, (0, 220, 0),
                    max(2, int(4 * s)), cv2.LINE_AA)
    cv2.circle(img, (mx, my), int(14 * s), (0, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (mx, my), int(3 * s), (0, 0, 0), -1, cv2.LINE_AA)
    if not face_ok:
        put_centered(img, "face not detected - face the camera, check lighting", H - int(40 * s),
                     0.8 * s, (60, 60, 255), 2)
    return img


def capture_point(ui, cam, tracker, mx, my, title, subtitle, idx, total):
    t0 = time.perf_counter()
    got = []
    while True:
        frame = read_frame(cam)
        if frame is None:
            continue
        now = time.perf_counter()
        el = now - t0
        obs = tracker.process(frame, now)
        ok = obs is not None and obs.gaze_fresh and max(obs.blink_l, obs.blink_r) < 0.4
        if el >= SETTLE_S and ok:
            got.append(obs.features())
        ui.show(draw_point_screen(ui.blank(), mx, my, title, subtitle, idx, total, el, obs is not None))
        if el >= SETTLE_S + CAPTURE_S:
            break
    return np.median(np.array(got), axis=0) if len(got) >= 8 else None


def calibrate_pose(ui, cam, tracker, W, H):
    print("Calibration 1/2: follow the dots. Point your NOSE at each dot and keep your eyes on it.")
    inset = int(40 * H / 1080.0) + 10
    F, P = [], []
    for i, (fx, fy, label) in enumerate(CAL_POINTS):
        px, py = fx * (W - 1), fy * (H - 1)
        mx, my = int(clamp(px, inset, W - 1 - inset)), int(clamp(py, inset, H - 1 - inset))
        sub = ("Your webcam is right above this dot. Sit the way you normally will."
               if i == 0 else "Turn / tilt your head so your nose points at the dot, eyes on it.")
        feat = None
        for _ in range(3):
            feat = capture_point(ui, cam, tracker, mx, my, label, sub, i, len(CAL_POINTS))
            if feat is not None:
                break
        if feat is None:
            raise RuntimeError("Could not see your face during calibration.")
        F.append(feat)
        P.append([px, py])
    F, P = np.array(F), np.array(P)
    span = F[:, :2].max(0) - F[:, :2].min(0)
    if span.min() < 2.0:
        print(f"WARNING: you moved your head very little (yaw span {span[0]:.1f}\u00b0, pitch span "
              f"{span[1]:.1f}\u00b0). Re-run with --recalibrate and turn your head more.")
    return F, P


def calibrate_wink(ui, cam, tracker):
    print("Calibration 2/2: wink test.")
    tracker.swap_eyes = False                    # collect RAW eyeBlinkLeft / eyeBlinkRight
    steps = [("rest", "Relax - keep BOTH eyes open", 1.8, 0.5),
             ("left", "Close ONLY your LEFT eye and hold it", 3.4, 1.4),
             ("right", "Close ONLY your RIGHT eye and hold it", 3.4, 1.4)]
    data = {}
    for key, text, dur, skip in steps:
        t0 = time.perf_counter()
        rows = []
        while True:
            frame = read_frame(cam)
            if frame is None:
                continue
            now = time.perf_counter()
            el = now - t0
            obs = tracker.process(frame, now)
            img = ui.blank()
            s = ui.H / 1080.0
            put_centered(img, text, int(ui.H * 0.4), 1.4 * s, (255, 255, 255), max(1, int(2 * s)))
            put_centered(img, "(winks are detected from a few eye-closing measurements)"
                         if key == "rest" else "Get ready... then hold it until the bar fills",
                         int(ui.H * 0.4) + int(50 * s), 0.7 * s, (190, 190, 190), 1)
            bw = int(500 * s)
            x0, y0 = (ui.W - bw) // 2, int(ui.H * 0.55)
            cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + int(14 * s)), (90, 90, 90), 1)
            cv2.rectangle(img, (x0, y0), (x0 + int(bw * clamp(el / dur, 0, 1)), y0 + int(14 * s)),
                          (0, 220, 0), -1)
            if obs is None:
                put_centered(img, "face not detected", int(ui.H * 0.7), 0.9 * s, (60, 60, 255), 2)
            elif el >= skip:
                rows.append([obs.blink_l, obs.blink_r])
            ui.show(img)
            if el >= dur:
                break
        if len(rows) < 8:
            raise RuntimeError("Could not see your face during wink calibration.")
        data[key] = np.array(rows)
    res = analyse_wink(data["rest"], data["left"], data["right"])
    tracker.swap_eyes = res["swap"]
    if not res["ok"]:
        print("  Could not clearly separate your two eyes - using defaults. Try --recalibrate in good light.")
        res = copy.deepcopy(DEFAULT_WINK)
    for side in ("left", "right"):
        if not res[side]["enabled"]:
            print(f"  Your {side} wink wasn't distinct enough from a blink -> {side} wink DISABLED.")
    return {k: res[k] for k in ("swap", "left", "right")}


# --------------------------------------------------------------------------- #
# profile persistence
# --------------------------------------------------------------------------- #
def load_profile(W, H) -> dict:
    try:
        d = json.loads(PROFILE_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}
    if d.get("screen") != [W, H]:
        print("Screen size changed since last calibration - recalibrating.")
        return {}
    return d


def save_profile(d: dict):
    APP_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(d, indent=1))


def ensure_calibration(profile, cam, tracker, W, H, force_pose=False, force_wink=False):
    need_pose = force_pose or "F" not in profile
    need_wink = force_wink or "wink" not in profile
    if not (need_pose or need_wink):
        return profile
    ui = Fullscreen(W, H)
    try:
        if need_pose:
            F, P = calibrate_pose(ui, cam, tracker, W, H)
            profile["F"], profile["P"] = F.tolist(), P.tolist()
        if need_wink:
            profile["wink"] = calibrate_wink(ui, cam, tracker)
    finally:
        ui.close()
    profile["screen"] = [W, H]
    save_profile(profile)
    return profile


# --------------------------------------------------------------------------- #
# mouse, hotkeys, preview
# --------------------------------------------------------------------------- #
class Mouse:
    def __init__(self, dry=False):
        try:
            import pyautogui
        except Exception as e:  # noqa: BLE001
            sys.exit(f"pyautogui is required ({e}). pip install pyautogui")
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.pg, self.dry = pyautogui, dry
        size = pyautogui.size()
        self.W, self.H = int(size[0]), int(size[1])
        pos = pyautogui.position()
        self.x, self.y = int(pos[0]), int(pos[1])

    def move(self, x, y):
        self.x, self.y = int(x), int(y)
        if not self.dry:
            self.pg.moveTo(self.x, self.y, _pause=False)

    def click(self, button="left"):
        if self.dry:
            print(f"[DRY RUN] Click {button} at ({self.x}, {self.y})")
            return
        self.pg.click(button=button, _pause=False)

    def down(self, button="left"):
        if not self.dry:
            self.pg.mouseDown(button=button, _pause=False)

    def up(self, button="left"):
        if not self.dry:
            self.pg.mouseUp(button=button, _pause=False)


class Controls:
    def __init__(self):
        self._pending: set = set()

    def push(self, cmd):
        self._pending.add(cmd)

    def take(self, cmd) -> bool:
        if cmd in self._pending:
            self._pending.discard(cmd)
            return True
        return False


def start_hotkeys(ctl: Controls):
    try:
        from pynput import keyboard
    except Exception:  # noqa: BLE001
        print("(pynput not installed - global F7-F10 hotkeys disabled; use the preview window keys)")
        return None
    K = keyboard.Key
    table = {K.f7: "reset", K.f8: "pause", K.f9: "recal", K.f10: "quit"}

    def on_press(key):
        cmd = table.get(key)
        if cmd is None and hasattr(key, "char") and key.char:
            cmd = {"c": "reset", "r": "recal"}.get(key.char.lower())
        if cmd:
            ctl.push(cmd)

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


@dataclass
class Hud:
    state: str = "TRACKING"
    control: str = "absolute"
    input: str = "head"
    pos: tuple = (0, 0)
    screen: tuple = (1920, 1080)
    flash: str = ""
    fps: float = 0.0
    dragging: bool = False
    right_swap: bool = False


STATE_COLOURS = {"TRACKING": (0, 220, 0), "PAUSED": (0, 165, 255),
                 "NO FACE": (0, 0, 255), "FROZEN": (0, 255, 255)}
KEYMAP = {27: "quit", ord("q"): "quit", ord("p"): "pause", ord(" "): "pause", 32: "pause",
          ord("c"): "reset", ord("C"): "reset", ord("w"): "wink",
          ord("r"): "recal", ord("R"): "recal",
          ord("m"): "control", ord("i"): "input"}


class Preview:
    NAME = "Eye Mouse Plus"

    def __init__(self, width=480, screen=None):
        self.width, self.screen, self._placed = width, screen, False
        cv2.namedWindow(self.NAME, cv2.WINDOW_AUTOSIZE)

    def render(self, frame, obs: Optional[Obs], hud: Hud):
        img = cv2.flip(frame, 1)                                   # mirror: feels natural
        h, w = img.shape[:2]
        img = cv2.resize(img, (self.width, int(h * self.width / w)))
        H2, W2 = img.shape[:2]
        if obs is not None:
            for i in PREVIEW_POINTS:
                cv2.circle(img, (int((1 - obs.pts[i, 0]) * W2), int(obs.pts[i, 1] * H2)), 2,
                           (0, 255, 255), -1, cv2.LINE_AA)
        colour = STATE_COLOURS.get(hud.state, (255, 255, 255))
        state_label = f"{hud.state} (SPACE: toggle)" if hud.state in ("TRACKING", "PAUSED") else hud.state
        if hud.right_swap:
            state_label += " [R-SWAP]"
        put_text(img, state_label + ("  [DRAG]" if hud.dragging else ""), (8, 22), 0.65, colour, 2)
        put_text(img, f"{hud.control} | {hud.input} | {hud.fps:4.0f} fps", (8, 44), 0.5)
        if obs is not None:
            put_text(img, f"yaw {obs.yaw:+5.1f}  pitch {obs.pitch:+5.1f}", (8, 64), 0.5)
            # Label bars according to --right swap:
            bars = (("R*", obs.blink_r), ("L", obs.blink_l)) if hud.right_swap else (("L*", obs.blink_l), ("R", obs.blink_r))
            for k, (name, v) in enumerate(bars):
                x = 10 + k * 34
                cv2.rectangle(img, (x, H2 - 70), (x + 22, H2 - 22), (255, 255, 255), 1)
                cv2.rectangle(img, (x, H2 - 22 - int(48 * clamp(v, 0, 1))), (x + 22, H2 - 22),
                              (0, 200, 255), -1)
                put_text(img, name, (x + 4, H2 - 6), 0.5)
        if hud.flash:
            put_centered(img, hud.flash, int(H2 * 0.5), 1.0, (0, 255, 0), 2)
        mw, mh = 128, 80                                           # mini screen map
        x0, y0 = W2 - mw - 8, H2 - mh - 8
        cv2.rectangle(img, (x0, y0), (x0 + mw, y0 + mh), (255, 255, 255), 1)
        cv2.rectangle(img, (x0 + mw // 2 - 5, y0 - 4), (x0 + mw // 2 + 5, y0), (0, 200, 255), -1)
        sw, sh = hud.screen
        cv2.circle(img, (x0 + int(hud.pos[0] / max(sw - 1, 1) * mw),
                         y0 + int(hud.pos[1] / max(sh - 1, 1) * mh)), 4, (0, 0, 255), -1)
        put_text(img, "c: centre | r: recalibrate | wink: click at 7/14/21 frames",
                 (8, 84), 0.36)
        return img

    def show(self, img) -> int:
        cv2.imshow(self.NAME, img)
        if not self._placed and self.screen:
            self._placed = True
            try:   # park it bottom-right so it covers as little as possible
                cv2.moveWindow(self.NAME, max(self.screen[0] - img.shape[1] - 20, 0),
                               max(self.screen[1] - img.shape[0] - 90, 0))
            except Exception:  # noqa: BLE001
                pass
        return cv2.waitKey(1) & 0xFF


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Head/eye cursor control with frame-counted wink clicking",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--cam-width", type=int, default=1280)
    p.add_argument("--cam-height", type=int, default=720)
    p.add_argument("--input", choices=INPUT_ORDER + ["auto"], default="head",
                   help="what drives the cursor: head pose, iris gaze, both, or best-by-calibration")
    p.add_argument("--control", choices=CONTROL_ORDER, default="absolute",
                   help="absolute: pose = screen position; joystick: pose offset = cursor speed")
    p.add_argument("--gain", type=float, default=1.0, help=">1 reaches the edges with smaller movements")
    p.add_argument("--smoothing", type=float, default=0.6, help="0 = raw/jittery .. 1 = very smooth")
    p.add_argument("--deadband", type=float, default=3.0, help="absolute mode: ignore moves under N px")
    p.add_argument("--speed", type=float, default=1.0, help="joystick: screens per second at full tilt")
    p.add_argument("--deadzone", type=float, default=0.12, help="joystick: neutral zone (0..0.9)")
    p.add_argument("--long-hold", type=float, default=0.18,
                   help="seconds the secondary eye must stay closed for a single right click")
    p.add_argument("--right", action="store_true",
                   help="swap eyes: RIGHT-eye wink = left click, LEFT-eye wink = right click")
    p.add_argument("--pause-hold", type=float, default=1.2,
                   help="seconds both eyes must be shut to toggle pause")
    p.add_argument("--no-right-click", action="store_true", help="disable right-wink = right click")
    p.add_argument("--mouth-drag", action="store_true", help="open mouth ~0.3 s to toggle drag-lock")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="track and show everything but never touch the mouse")
    p.add_argument("--paused", "--start-paused", dest="start_paused", action="store_true",
                   help="start with cursor control paused (press SPACE to activate)")
    return p.parse_args(argv)


def make_wink(profile, args) -> WinkDetector:
    wp = copy.deepcopy(profile["wink"])
    if args.right:                                  # swap roles AND per-eye thresholds
        wp["left"], wp["right"] = wp["right"], wp["left"]
    if args.no_right_click:
        wp["right"]["enabled"] = False
    return WinkDetector(wp, long_hold=args.long_hold, pause_hold=args.pause_hold)


def run(args):
    check_platform()
    mouse = Mouse(dry=args.dry_run)
    W, H = mouse.W, mouse.H
    model = ensure_model()
    cam = open_camera(args.camera, args.cam_width, args.cam_height)
    tracker = FaceTracker(model)
    ctl = Controls()
    hotkeys = start_hotkeys(ctl)
    preview = None
    dragging = False

    try:
        profile = {} if args.recalibrate else load_profile(W, H)
        try:
            profile = ensure_calibration(profile, cam, tracker, W, H)
        except CalibrationAborted:
            print("Calibration aborted.")
            return
        tracker.swap_eyes = profile["wink"]["swap"]
        F, P = np.array(profile["F"]), np.array(profile["P"])
        mapper, input_name = build_mapper(F, P, W, H, args.input)
        engine = CursorEngine(W, H, mapper, args.control, args.gain, args.smoothing,
                              args.speed, args.deadzone)
        engine.pos[:] = (mouse.x, mouse.y)
        preview = None if args.no_preview else Preview(screen=(W, H))
        wink = make_wink(profile, args)
        mouth = MouthDrag() if args.mouth_drag else None

        paused, was_frozen = args.start_paused, False
        last_sent = None
        flash_text, flash_until = "", 0.0
        lost_since = None
        fps, prev_t = 0.0, time.perf_counter()
        hud = Hud(screen=(W, H), right_swap=args.right)
        init_state = "PAUSED" if paused else "ACTIVE"
        swap_str = " [SWAP: Right=click, Left=right-click]" if args.right else ""
        print(f"Running. Cursor control is {init_state}{swap_str}. Press 'c' / F7 to centre, 'r' / F9 to recalibrate, SPACE to toggle pause.")

        while True:
            frame = read_frame(cam)
            if frame is None:
                continue
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-3)
            prev_t = now
            obs = tracker.process(frame, now)

            # ---------------- commands (hotkeys + preview keys) -----------------
            if ctl.take("quit"):
                break
            if ctl.take("pause"):
                paused = not paused
                beep(not paused)
                if not paused:
                    engine.release(now)
                status_str = "PAUSED" if paused else "ACTIVE"
                print(f"[CURSOR CONTROL] {status_str} (Press SPACE, 'p', or F8 to toggle)")
            if ctl.take("control"):
                engine.control = CONTROL_ORDER[(CONTROL_ORDER.index(engine.control) + 1) % 2]
                engine.filt.reset()
            if ctl.take("input"):
                input_name = INPUT_ORDER[(INPUT_ORDER.index(input_name) + 1) % 3]
                engine.mapper = Mapper().fit(F, P, INPUT_SETS[input_name])
                engine.filt.reset()
            if ctl.take("reset") or ctl.take("recenter"):
                if obs is not None:
                    engine.recenter(obs.features())
                else:
                    engine.pos[:] = engine.centre
                    engine.bubble[:] = engine.centre
                    engine.filt.reset()
                mouse.move(int(engine.pos[0]), int(engine.pos[1]))
                last_sent = (int(engine.pos[0]), int(engine.pos[1]))
                flash_text, flash_until = "RESET / RECENTER", now + 0.8
                beep(high=True)
                print(f"[RESET] Cursor centered at ({int(engine.pos[0])}, {int(engine.pos[1])})")

            for cmd in ("recal", "wink"):
                if ctl.take(cmd):
                    try:
                        profile = ensure_calibration(profile, cam, tracker, W, H,
                                                     force_pose=(cmd == "recal"),
                                                     force_wink=(cmd == "wink"))
                        tracker.swap_eyes = profile["wink"]["swap"]
                        F, P = np.array(profile["F"]), np.array(profile["P"])
                        engine.mapper = build_mapper(F, P, W, H, input_name)[0]
                        engine.filt.reset()
                        wink = make_wink(profile, args)
                    except CalibrationAborted:
                        print("Calibration aborted - keeping the previous one.")
                    except RuntimeError as e:
                        print(e)
                    obs = None

            # ---------------- cursor + gestures -----------------------------------
            state = "TRACKING"
            frozen = False
            if obs is None:
                wink.clear()
                state = "NO FACE"
                lost_since = lost_since or now
                if now - lost_since > 1.0:
                    engine.filt.reset()
            else:
                lost_since = None
                eye_a, eye_b = (obs.blink_r, obs.blink_l) if args.right else (obs.blink_l, obs.blink_r)
                events = wink.update(now, eye_a, eye_b)
                if any(ev[0] == "toggle_pause" for ev in events):
                    paused = not paused
                    beep(not paused)
                    if not paused:
                        engine.release(now)
                    status_str = "PAUSED" if paused else "ACTIVE"
                    print(f"[CURSOR CONTROL] {status_str} (both eyes closed ~1.2 s)")
                if paused:
                    wink.clear()
                frozen = wink.frozen(now)

                if not paused:
                    if frozen:
                        if not was_frozen:
                            engine.rewind(now - 0.10)
                            mouse.move(*engine.pos)
                            last_sent = (int(engine.pos[0]), int(engine.pos[1]))
                    else:
                        if was_frozen:
                            engine.release(now)
                        engine.update(now, obs.features())
                        x, y = int(round(engine.pos[0])), int(round(engine.pos[1]))
                        db = args.deadband if engine.control == "absolute" else 1.0
                        if last_sent is None or math.hypot(x - last_sent[0], y - last_sent[1]) >= db:
                            mouse.move(x, y)
                            last_sent = (x, y)

                    for side, _ in events:
                        if side in ("left", "right"):
                            mouse.click(side)
                            click_name = "CLICK" if side == "left" else "RIGHT CLICK"
                            if args.right:
                                click_name = "RIGHT EYE CLICK" if side == "left" else "LEFT EYE CLICK"
                            flash_text, flash_until = click_name, now + 0.6

                    if mouth is not None and mouth.update(now, obs.jaw):
                        dragging = not dragging
                        (mouse.down if dragging else mouse.up)()
                        flash_text, flash_until = ("DRAG ON" if dragging else "DRAG OFF"), now + 0.6
                was_frozen = frozen

            # ---------------- preview ---------------------------------------------
            if preview is not None:
                if paused:
                    state = "PAUSED"
                elif frozen:
                    state = "FROZEN"
                hud.state, hud.control, hud.input = state, engine.control, input_name
                hud.pos = (mouse.x, mouse.y)
                hud.fps, hud.dragging = fps, dragging
                hud.flash = flash_text if now < flash_until else ""
                hud.right_swap = args.right
                key = preview.show(preview.render(frame, obs, hud))
                if key in KEYMAP:
                    ctl.push(KEYMAP[key])
    except KeyboardInterrupt:
        pass
    finally:
        if dragging:
            mouse.up()
        cam.release()
        cv2.destroyAllWindows()
        if hotkeys is not None:
            hotkeys.stop()


if __name__ == "__main__":
    run(parse_args())
