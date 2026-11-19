"""Desktop entry point used by the Windows shortcut."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from tkinter import messagebox

from pilot_setup import Preferences, show_setup
from pilot_startup import DESKTOP_LOCK, configure_wake, session_lock, wait_for_wake_microphone, wake_enabled

APP_DIR = Path.home() / ".eye_mouse"
SETUP_PATH = APP_DIR / "setup.json"


def read_preferences():
    try:
        data = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
        return Preferences(**data["preferences"])
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


def save_preferences(preferences):
    APP_DIR.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=APP_DIR, delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        temporary_path.write_text(json.dumps({
            "preferences": vars(preferences),
            "launch_flags": preferences.launch_flags(),
        }, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(SETUP_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    stdout, stderr = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w", encoding="utf-8") as quiet:
            sys.stdout = stdout or quiet
            sys.stderr = stderr or quiet
            try:
                if sys.argv[1:] == ["--disable-wake"]:
                    change_wake_setting(False)
                elif sys.argv[1:] == ["--wake-settings"]:
                    show_wake_settings()
                else:
                    with session_lock(DESKTOP_LOCK) as acquired:
                        if acquired:
                            wait_for_wake_microphone()
                            launch_desktop(start_paused="--start-paused" in sys.argv[1:])
            finally:
                sys.stdout, sys.stderr = stdout, stderr
    except (Exception, SystemExit) as error:
        messagebox.showerror("Pilot", str(error) or "Pilot could not start.")


def change_wake_setting(enabled):
    configure_wake(enabled)
    saved = read_preferences()
    if saved is not None:
        save_preferences(replace(saved, startup_listener=enabled))


def show_wake_settings():
    state = "on" if wake_enabled() else "off"
    enabled = messagebox.askyesnocancel(
        "Pilot Wake Listener",
        f"Starting the wake listener with Windows is currently {state}.\n\n"
        "Start it automatically when you log into Windows?\n"
        "Say “Pilot” to open the app while it’s listening.",
    )
    if enabled is not None:
        change_wake_setting(enabled)


def launch_desktop(*, start_paused=False):
    session_flags = ["--start-paused"] if start_paused else []
    saved = read_preferences()
    if saved is not None:
        from pilot import parse_args, run

        run(parse_args([*saved.launch_flags(), *session_flags]))
        return

    preferences = Preferences(startup_listener=wake_enabled())
    error_message = ""
    while True:
        preferences = show_setup(preferences, error_message)
        if preferences is None:
            return

        from pilot import parse_args, run

        completed = False

        def finish_setup():
            nonlocal completed
            try:
                configure_wake(preferences.startup_listener)
            except (OSError, RuntimeError) as error:
                messagebox.showwarning("Pilot Wake Listener", str(error))
                save_preferences(replace(preferences, startup_listener=wake_enabled()))
            else:
                save_preferences(preferences)
            completed = True

        try:
            flags = [*preferences.launch_flags(), *session_flags, "--recalibrate"]
            run(parse_args(flags), on_calibrated=finish_setup)
        except (RuntimeError, SystemExit, OSError) as error:
            if completed:
                raise
            error_message = str(error)
        else:
            if completed:
                return
            error_message = "Calibration was canceled. Start again when you’re ready."


if __name__ == "__main__":
    main()
