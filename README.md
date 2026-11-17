# Pilot

Pilot controls the Windows cursor with a webcam, wink clicks, and spoken commands. Say **“dictate”**, speak a phrase, and pause: local Whisper transcribes it and types the text into the focused window.

Camera and speech processing run on your computer. This is an experimental accessibility project; tracking and recognition still depend on calibration, lighting, microphone quality, and hardware.

## Setup

Use Windows, Python 3.12, and `uv`, with a webcam and microphone. Run from the repository root, outside WSL.

```powershell
git clone https://github.com/parm2006/Pilot.git
cd Pilot
uv sync
uv run python -c "from faster_whisper.utils import download_model; from pilot_dictation import MODEL_CACHE; download_model('base.en', cache_dir=str(MODEL_CACHE))"
```

The last command downloads the Whisper dictation weights into `models/whisper`. They load offline on subsequent launches. The face, hand, and Vosk speech models download on first use into `%USERPROFILE%\.eye_mouse`; that folder also stores calibration.

Optional: install and open Ollama, then download the text embedding model for fuzzy command matching:

```powershell
ollama pull all-minilm
```

Ollama matches recognized text to commands. Vosk recognizes commands; Whisper transcribes dictation. Audio is not sent to Ollama or a cloud service.

## Run

```powershell
uv run python -u pilot.py --paused --voice-verbose
```

Follow calibration if prompted. Cursor control starts paused; **F8** enables it and **F10** quits. Voice commands remain active while the cursor is paused.

To run without Ollama, add `--embed-backend none`. To inspect tracking, recognition, and dictation without sending mouse clicks, shortcuts, or text to other apps, use:

```powershell
uv run python -u pilot.py --dry-run --voice-verbose --embed-backend none
```

## Spoken controls

| Say | Action |
| --- | --- |
| “center cursor” | Recenter the cursor |
| “recalibrate” | Repeat tracking calibration |
| “calibrate winks” | Repeat wink calibration |
| “switch cursor mode” | Cycle absolute / joystick control |
| “switch tracking input” | Cycle head / eyes / hybrid input |
| “pause pilot” / “resume pilot” | Pause or resume cursor control |
| “pause listener” / “resume listener” | Pause or resume voice commands |
| “quit pilot” or “stop listener” | Exit the whole program |
| “dictate” | Capture and type one spoken phrase |

Say **“control” followed by one letter**, such as “control C” or “control V”, to send Ctrl+C or Ctrl+V to the focused app. A–Z are supported; arbitrary combinations such as “control shift T” are not. Named commands can still send their configured shortcuts.

Other commands include copy/paste, selection, navigation, Page Up/Down, browser shortcuts, mouse clicks, scrolling, and Windows media keys. The full list and aliases live in [commands.json](commands.json), which reloads when saved. Shortcuts act on the focused app; scroll commands act at the pointer. “Stop” stops scrolling; it does not quit Pilot.

## Dictation

1. Focus the text field you want to type into.
2. Say **“dictate”** and wait for the ready message in the terminal or the `DICTATING` status in the preview.
3. Speak your text. About **0.8 seconds of silence** ends capture.
4. Local Whisper transcribes the phrase, Pilot types it, then returns to command mode. Say “dictate” again for another phrase.

There is one microphone stream. Pilot routes incoming audio to either command recognition or dictation. Text appears after the phrase ends and decoding finishes; it is not word-by-word live transcription. Spoken command names inside dictation are treated as text.

The default model is English `base.en`, running on the CPU with int8 inference. Each phrase is capped at 60 seconds; waiting ten seconds without speech cancels capture. Change the pause threshold with `--dictation-gap 1.2` if natural pauses cut phrases short.

Raising a hand to the camera discards microphone input and cancels pending dictation. **F6** pauses voice and cancels dictation. If the foreground window changes before insertion, the transcript is printed but not typed into the new window.

## Keyboard and gesture backups

| Control | Action |
| --- | --- |
| F6 | Toggle voice listening |
| F7 | Recenter |
| F8 | Toggle cursor control |
| F9 | Recalibrate tracking |
| F10 | Quit |
| Preview: Space / P | Toggle cursor control |
| Preview: C / R / W | Recenter / tracking calibration / wink calibration |
| Preview: M / I / V | Cursor mode / tracking input / voice listening |
| Preview: Q / Esc | Quit |

Left wink performs the primary click; right wink performs a right click when enabled by calibration. `--right` swaps the eyes. Holding both eyes closed for about 1.2 seconds toggles cursor control. Bare C and R are only preview controls, so typing them in other apps does not recenter or recalibrate Pilot.

## Files and checks

- `pilot.py`: camera, tracking, calibration, cursor control, and preview.
- `pilot_voice.py`: microphone capture, command recognition, matching, and execution.
- `pilot_dictation.py`: speech/pause detection, local Whisper, and Unicode text insertion.
- `commands.json`: active command definitions and aliases.
- `finetune_embedder.py` and `paraphrases.json`: optional embedding-training tools; normal use does not require them. No trained weights are bundled.
- `old/`: earlier versions, including the snapshot taken before promoting the former `claude-test` version into the root.

```powershell
uv run python -m unittest discover -v
```

Automated checks cover command routing, dictation boundaries, cancellation, focus changes, and dry-run behavior. Synthesized speech and Unicode insertion have also been checked locally. Accuracy with a person's voice and real webcam calibration still requires hands-on testing.
