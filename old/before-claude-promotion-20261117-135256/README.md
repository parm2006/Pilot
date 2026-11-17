# Pilot

Pilot is an accessibility-focused desktop control tool that lets a user interact with their computer using eye movement, intentional blinks, and voice input.

The project is intended to explore hands-free computing rather than replace a conventional mouse and keyboard. Its first version should focus on a small, dependable set of interactions:

- Move the pointer based on the user’s gaze.
- Calibrate the camera to the user and screen.
- Use intentional eye gestures for click, double-click, and drag.
- Use voice commands or speech-to-text for typing and basic actions.
- Provide clear visual feedback about gaze position, calibration state, and detected gestures.
- Include pause, emergency-stop, and sensitivity controls so accidental input is easy to prevent.

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

