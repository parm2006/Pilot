# Pilot

Local, hands-free Windows cursor control using a webcam, wink clicks, spoken commands, and Whisper dictation.

## Setup

Requires Windows, Python 3.12, `uv`, a webcam, and a microphone. Run outside WSL.

```powershell
uv sync
uv run python -B -c "from faster_whisper.utils import download_model; from pilot_dictation import MODEL_CACHE; download_model('base.en', cache_dir=str(MODEL_CACHE))"
```

Models and calibration live under `%USERPROFILE%\.eye_mouse`, outside the project. Whisper uses `.eye_mouse\whisper`; the other models download on first use. Audio is processed locally.

Optional fuzzy command matching uses Ollama. Open Ollama and run `ollama pull all-minilm`, or add `--embed-backend none` when launching Pilot.

## Run

```powershell
uv run python -B -u pilot.py --paused --voice-verbose
```

Follow calibration if prompted. **F8** enables cursor control; **F10** quits. Add `--dry-run` to print actions without sending mouse clicks, keyboard shortcuts, or dictated text to other apps. `-B` prevents Python bytecode cache folders beside the source files.

## Voice commands

| Say | Action |
| --- | --- |
| “center cursor” | Recenter |
| “recalibrate” | Calibrate tracking |
| “calibrate winks” | Calibrate wink clicks |
| “switch cursor mode” | Cycle absolute / joystick control |
| “switch tracking input” | Cycle head / eyes / hybrid input |
| “pause pilot” / “resume pilot” | Pause / resume cursor control |
| “pause listener” / “resume listener” | Pause / resume voice commands |
| “quit pilot” or “stop listener” | Exit Pilot |
| “dictate” | Capture one phrase for local Whisper dictation |

“Control” plus any single letter sends that shortcut to the focused app: “control C”, “control V”, “control A”, and so on. Arbitrary multi-modifier shortcuts are not supported. More named commands, aliases, clicks, scrolling, Page Up/Down, and media keys are defined in [commands.json](commands.json), which reloads when saved. “Stop” stops scrolling only.

## Dictation and backups

Focus a text field, say **“dictate”**, wait for `DICTATING` in the preview or the terminal's ready message, then speak. A pause of about **0.8 seconds** ends capture; Whisper transcribes and inserts the phrase, then Pilot returns to command mode. Say “dictate” again for another phrase. Text appears after decoding, not word by word. Adjust the pause using `--dictation-gap 1.2`.

Dictation uses English `base.en` on the CPU, with a 60-second phrase limit and a ten-second timeout if no speech starts. Spoken command names during dictation become text. Raising a hand mutes microphone input and cancels pending dictation. Text is not inserted if the foreground window changed before insertion.

**F6** toggles voice and cancels dictation; **F7** recenters; **F9** recalibrates. In the preview, Space/P pauses, C recenters, R recalibrates, W calibrates winks, M switches cursor mode, I switches input, V toggles voice, and Q/Esc quits. Bare C/R are not global hotkeys.

Left wink performs a primary click; right wink right-clicks when enabled by calibration. `--right` swaps the eyes. Holding both eyes closed for about 1.2 seconds toggles cursor control.

## Code and checks

All source and test files are directly in the root:

- `pilot.py`: camera, tracking, calibration, cursor, preview, and hotkeys.
- `pilot_voice.py`: microphone, recognition, command matching, and execution.
- `pilot_dictation.py`: pause detection, Whisper, and Unicode text insertion.
- `commands.json`: command definitions.
- `test_dictation.py`, `test_voice_logging.py`: automated checks.

`pyproject.toml`, `uv.lock`, and `.python-version` define the environment. `.venv` is the installed Python environment; `.git` stores version history.

```powershell
uv run python -B -m unittest discover -v
```

Checks cover command routing, dictation boundaries, cancellation, focus changes, and dry-run behavior. Voice accuracy and webcam calibration still need hands-on testing.
