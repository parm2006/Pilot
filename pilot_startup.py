"""Windows session locks and optional registration of the wake listener."""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import winreg

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "PilotWakeListener"
DESKTOP_LOCK = r"Local\Pilot.Desktop"
WAKE_MIC_LOCK = r"Local\Pilot.WakeMicrophone"
WAKE_STOP_EVENT = r"Local\Pilot.WakeStop"

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
kernel.CreateMutexW.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
kernel.WaitForSingleObject.restype = wintypes.DWORD
kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
kernel.OpenEventW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
kernel.OpenEventW.restype = wintypes.HANDLE
kernel.SetEvent.argtypes = (wintypes.HANDLE,)


@contextmanager
def session_lock(name, timeout_ms=0):
    handle = kernel.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        result = kernel.WaitForSingleObject(handle, timeout_ms)
        if result not in (0, 0x80, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = result != 258
        yield acquired
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def wait_for_wake_microphone():
    with session_lock(WAKE_MIC_LOCK, timeout_ms=6500) as free:
        if not free:
            raise RuntimeError("Pilot’s wake listener is still releasing the microphone. Try again shortly.")


def wake_command():
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    listener = Path(__file__).resolve().with_name("pilot_listener.ps1")
    return [str(powershell), "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
            "-ExecutionPolicy", "RemoteSigned", "-File", str(listener)]


def wake_enabled():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, RUN_VALUE)
        return True
    except FileNotFoundError:
        return False


def disable_wake():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, access=winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, RUN_VALUE)
    except FileNotFoundError:
        pass
    handle = kernel.OpenEventW(0x0002, False, WAKE_STOP_EVENT)  # EVENT_MODIFY_STATE
    if handle:
        try:
            if not kernel.SetEvent(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)


def configure_wake(enabled):
    if not enabled:
        disable_wake()
        return
    command = wake_command()
    startup_command = subprocess.list2cmdline([*command, "-Registered"])
    if len(startup_command) > 260:
        raise RuntimeError("Pilot’s install path is too long for Windows startup. Install to a shorter folder.")
    try:
        result = subprocess.run([*command, "-Check"], capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Windows Speech did not respond. Try enabling the wake listener again.") from error
    if result.returncode != 0:
        raise RuntimeError("The wake listener needs English Speech installed in Windows Settings.")
    was_registered = wake_enabled()
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, access=winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, startup_command)
    try:
        subprocess.Popen([*command, "-Registered"], creationflags=subprocess.CREATE_NO_WINDOW,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        if not was_registered:
            disable_wake()
        raise
