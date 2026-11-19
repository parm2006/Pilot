# Pilot

Pilot is an accessibility-focused desktop control tool that lets a user interact with their computer using eye movement, intentional blinks, and voice input.

The project is intended to explore hands-free computing rather than replace a conventional mouse and keyboard. Its first version should focus on a small, dependable set of interactions:

- Move the pointer based on the user’s gaze.
- Calibrate the camera to the user and screen.
- Use intentional eye gestures for click, double-click, and drag.
- Use voice commands or speech-to-text for typing and basic actions.
- Provide clear visual feedback about gaze position, calibration state, and detected gestures.
- Include pause, emergency-stop, and sensitivity controls so accidental input is easy to prevent.

## Windows desktop app

When a GitHub Release is available, run its `PilotSetup.exe` and leave **Open Pilot**
selected at the end of installation, or open **Pilot** from the Windows Start menu.
The first launch asks for a primary click eye, pointer tracking mode, and voice
preferences, then starts the screen-target and wink exercises. Setup is saved
after both exercises finish. The default camera and microphone are used.

On later launches, Pilot starts directly with the saved settings. Use **F7** to
recenter, **F9** to recalibrate, or **F10** to close Pilot. Closing the Pilot
window or saying **“stop,” “stop Pilot,”** or **“stop listener”** ends the
desktop session. Say **“stop scrolling”** to stop only scrolling. During
dictation, use **“stop Pilot”** or **“stop listener”** to end the session.

**Start wake listener with Windows** is off by default. Enable it in setup or
open **Pilot Wake Listener Settings** from the Start menu. The wake listener
requires an installed English Windows Speech recognizer and a default
microphone. It runs locally and only listens for **“Pilot,” “start Pilot,”**
or **“start listener.”** It releases the microphone while Pilot is open and
resumes after the desktop session ends. Turn off the setting to stop the wake
listener itself. When the setting is off, open Pilot from the Start menu to
start the desktop session and its voice commands.

From a source checkout, run `uv run pilot_app.py` to open the same desktop app.
The desktop launcher saves preferences and launch flags in
`%LOCALAPPDATA%Pilotsetup.json` on Windows.
First use can download the tracking and voice command models. Whisper dictation
uses a separate locally cached model.

The Windows installer workflow builds a downloadable artifact when run manually.
A version tag that matches `pyproject.toml` publishes the installer to GitHub Releases.

## Initial goals

1. Build a webcam-based gaze estimator.
2. Map gaze coordinates to screen coordinates through a calibration flow.
3. Smooth pointer movement without making the pointer feel sluggish.
4. Distinguish intentional blinks from normal blinking well enough for basic clicking.
5. Add push-to-talk or wake-word speech input for typing.
6. Keep processing local by default and make camera/microphone use explicit.

## Deliberate non-goals for the first version

- Full facial-expression recognition.
- Medical or clinical eye-tracking claims.
- Perfect gaze accuracy across every lighting condition, camera, or user.
- Replacing all keyboard shortcuts and mouse interactions immediately.
- Cloud processing of camera or microphone data by default.

## Suggested milestones

### Milestone 1: Camera and landmarks

Display the webcam feed, detect a face and eyes, and draw landmarks and diagnostics on screen.

### Milestone 2: Calibration and gaze mapping

Guide the user through screen targets, collect gaze features, and fit a mapping from eye/face features to screen coordinates.

### Milestone 3: Pointer control

Add smoothing, dead zones, sensitivity settings, and a pause shortcut.

### Milestone 4: Intentional gestures

Add blink-based click and dwell-based click, with safeguards against accidental activation.

### Milestone 5: Voice input

Add push-to-talk speech-to-text and a small command vocabulary for actions such as click, pause, and scroll.

### Milestone 6: Usability and accessibility

Test with different lighting, glasses, camera positions, and users. Document limitations and add configuration profiles.

## Privacy and safety principles

- Process camera and microphone streams locally whenever practical.
- Do not save raw camera or microphone data unless the user explicitly enables recording.
- Make active tracking visible and easy to pause.
- Require deliberate confirmation for destructive actions.
- Treat this as an assistive software experiment, not a medical device.

## Palm gesture for voice muting

Hold either hand in view with your palm facing the camera and all five fingers
extended and spread apart to mute voice input. Lower your hand or relax the
gesture to resume listening after about 0.7 seconds. A fist, pointing finger,
fingers held together, or the back or edge of a hand does not qualify as the
mute gesture. Tracking and wink controls continue while voice input is muted.

The gesture must be detected on two consecutive hand checks to avoid brief
false detections. Keep the whole hand in frame. Use `--no-hand-mute` to disable
the gesture. This ignores microphone audio in Pilot; it does not mute the
Windows microphone for other applications.
