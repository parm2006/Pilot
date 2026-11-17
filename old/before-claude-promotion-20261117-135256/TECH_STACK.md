# Pilot Tech Stack

## Recommended language and runtime

- **Python 3.12+** — rapid experimentation and strong computer-vision ecosystem.
- **uv** — fast project and virtual-environment management.
- **Ruff** — linting and formatting.
- **pytest** — automated tests for calibration, smoothing, gesture detection, and command parsing.

## Computer vision and gaze estimation

- **OpenCV** — webcam capture, image processing, coordinate transforms, and display utilities.
- **MediaPipe Face Mesh / Face Landmarker** — face and eye landmark detection.
- **NumPy** — numerical operations and feature processing.
- **scikit-learn** — optional calibration models such as linear regression, ridge regression, or a lightweight polynomial model.

Start with a simple feature-based calibration model before attempting a neural-network gaze estimator.

## Desktop control

- **PyAutoGUI** — cross-platform pointer movement, clicks, keyboard input, and scrolling.
- **pynput** — optional low-level input hooks and global pause/resume shortcuts.

Keep all operating-system input actions behind a small adapter module so the vision system can be tested without moving the real pointer.

## Speech input

Recommended order:

1. **Local Whisper implementation** such as `faster-whisper` for privacy and offline use.
2. **sounddevice** or **PyAudio** for microphone capture.
3. A small local command parser for commands such as `click`, `double click`, `scroll down`, `pause`, and `type ...`.

An online speech provider can be added later behind the same interface, but it should not be required for the first milestone.

## User interface

- **PySide6 / Qt** — recommended for the calibration window, settings panel, status indicators, and accessibility controls.
- **Qt signals and timers** — useful for separating camera processing from UI updates.

Avoid building the first interface entirely with OpenCV windows; they are useful for diagnostics but limiting for a real settings and calibration experience.

## Suggested project structure

```text
Pilot/
├── README.md
├── TECH_STACK.md
├── pyproject.toml
├── src/
│   └── pilot/
│       ├── camera.py
│       ├── landmarks.py
│       ├── calibration.py
│       ├── gaze.py
│       ├── smoothing.py
│       ├── gestures.py
│       ├── pointer_control.py
│       ├── speech.py
│       ├── commands.py
│       ├── settings.py
│       └── ui/
└── tests/
```

## Core engineering approach

- Keep camera capture, landmark detection, gaze estimation, gesture recognition, speech recognition, and desktop control as separate modules.
- Use typed data objects for frames, gaze points, gestures, and commands.
- Add replayable recorded landmark data for testing without a live webcam.
- Log performance metrics such as camera FPS, inference time, and pointer latency.
- Make thresholds and smoothing parameters configurable rather than hard-coded.
