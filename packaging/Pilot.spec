import os
from pathlib import Path
import subprocess

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(SPECPATH).parent
datas = [(str(root / name), ".") for name in ("commands.json", "pilot_listener.ps1")]
for package in ("mediapipe", "faster_whisper"):
    datas += collect_data_files(package)

binaries = []
for package in ("vosk", "ctranslate2", "_sounddevice_data"):
    binaries += collect_dynamic_libs(package)

# Ship the app-local VC runtime so native tracking DLLs can load on a fresh PC.
vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
crt_paths = subprocess.check_output([
    str(vswhere), "-latest", "-products", "*", "-find",
    r"VC\Redist\MSVC\**\x64\Microsoft.VC*.CRT\concrt140.dll",
], text=True).splitlines()
crt_dirs = [Path(path).parent for path in crt_paths if "onecore" not in Path(path).parts]
if not crt_dirs:
    raise SystemExit("Install the Visual Studio C++ build tools to bundle the x64 VC runtime.")
binaries += [(str(dll), ".") for dll in crt_dirs[-1].glob("*.dll")]

a = Analysis(
    [str(root / "pilot_app.py")],
    pathex=[str(root)],
    datas=datas,
    binaries=binaries,
    hookspath=[str(root / "packaging" / "hooks")],
    hiddenimports=["pynput.keyboard._win32", "pynput.mouse._win32"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Pilot", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Pilot")
