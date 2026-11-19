import os
from pathlib import Path
import subprocess
import tomllib

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
)

root = Path(SPECPATH).parent
version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
version_parts = (*map(int, version.split(".")), 0)
resource = VSVersionInfo(
    ffi=FixedFileInfo(filevers=version_parts, prodvers=version_parts, fileType=1),
    kids=[
        StringFileInfo([StringTable("040904B0", [
            StringStruct("FileDescription", "Pilot"),
            StringStruct("FileVersion", version),
            StringStruct("InternalName", "Pilot"),
            StringStruct("OriginalFilename", "Pilot.exe"),
            StringStruct("ProductName", "Pilot"),
            StringStruct("ProductVersion", version),
        ])]),
        VarFileInfo([VarStruct("Translation", [1033, 1200])]),
    ],
)
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
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Pilot", console=False, version=resource)
coll = COLLECT(exe, a.binaries, a.datas, name="Pilot")
