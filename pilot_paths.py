"""Per-user Pilot data and compatibility with earlier installations."""

import os
from pathlib import Path

APP_DIR = Path(os.environ["LOCALAPPDATA"]) / "Pilot" if "LOCALAPPDATA" in os.environ else Path.home() / ".pilot"
LEGACY_DIR = Path.home() / ".eye_mouse"


def cached_data(name):
    path = APP_DIR / name
    legacy = LEGACY_DIR / name
    return legacy if not path.exists() and legacy.exists() else path


def migrate_settings():
    # Keep the old files available to earlier builds and avoid copying large models.
    for name in ("setup.json", "profile.json"):
        legacy, current = LEGACY_DIR / name, APP_DIR / name
        if legacy.is_file() and not current.exists():
            contents = legacy.read_bytes()
            APP_DIR.mkdir(parents=True, exist_ok=True)
            try:
                with current.open("xb") as stream:
                    stream.write(contents)
            except FileExistsError:
                pass
