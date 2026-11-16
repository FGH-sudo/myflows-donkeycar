"""Start the Udacity simulator and put it into autonomous mode with nobody at the mouse (Windows).

The simulator build has no command-line switch for any of this. It opens Unity's configuration dialog, then a
menu on which a track and "Autonomous Mode" have to be clicked, and an episode can only be restarted by going
back to that menu with Escape. The launcher does those steps through the window system:

* the dialog is an ordinary Windows dialog: resolution, quality and "Windowed" are set and "Play!" is pressed by
  messages to its controls, without focus or mouse;
* Escape reaches the game window as a posted key message;
* the menu buttons are Unity UI, which reads the real cursor position: the cursor is put on the button for a
  moment, the click is posted to the window, and the cursor goes back to where it was.

Whether a click arrived cannot be read from the window. The caller sees it on the wire: the simulator connects
to port 4567 when the scene has started (``apps.sim.udacity.session`` retries the click until it does).

On other systems start the simulator by hand, choose Autonomous Mode and use ``--sim remote``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
_BUILDS = {"win32": ("UdacitySimWin", "Default Windows desktop 64-bit.exe")}
ENVIRONMENT_VARIABLE = "UDACITY_SIM_PATH"
# Centres of the menu buttons as fractions of the window's client area (the menu scales with the window).
MODE_BUTTON = (.5, .63)
TRACK_BUTTONS = {"lake": (.352, .83), "jungle": (.648, .83)}
RESOLUTION, QUALITY = "640 x 480", "Fastest"
DIALOG_RESOLUTION, DIALOG_QUALITY, DIALOG_WINDOWED, DIALOG_PLAY = 1010, 1011, 1009, 1


class LauncherError(RuntimeError):
    pass


def default_executable() -> Path:
    override = os.environ.get(ENVIRONMENT_VARIABLE)
    if override:
        return Path(override)
    build = _BUILDS.get(sys.platform)
    if build is None:
        raise LauncherError("the simulator can only be started automatically on Windows: start it yourself, "
                            "choose Autonomous Mode and pass --sim remote")
    return ROOT.joinpath(*build)


class _Windows:
    """The few user32 calls the launcher needs, declared once."""

    WM_COMMAND, WM_KEYDOWN, WM_KEYUP = 0x0111, 0x0100, 0x0101
    WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP = 0x0200, 0x0201, 0x0202
    BM_GETCHECK, BM_CLICK = 0x00F0, 0x00F5
    CB_FINDSTRINGEXACT, CB_SETCURSEL, CBN_SELCHANGE = 0x0158, 0x014E, 1
    VK_ESCAPE, MK_LBUTTON = 0x1B, 0x0001
    SW_RESTORE = 9

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.ctypes, self.wintypes = ctypes, wintypes
        user32 = self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        try:  # report window rectangles in real pixels, whatever the display scaling
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()
        self.callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows.argtypes = [self.callback, wintypes.LPARAM]
        user32.EnumChildWindows.argtypes = [wintypes.HWND, self.callback, wintypes.LPARAM]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.GetDlgCtrlID.argtypes = [wintypes.HWND]
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        for function in (user32.SendMessageW, user32.PostMessageW):
            function.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.SendMessageW.restype = ctypes.c_ssize_t

    def class_name(self, hwnd) -> str:
        buffer = self.ctypes.create_unicode_buffer(256)
        self.user32.GetClassNameW(hwnd, buffer, 256)
        return buffer.value

    def windows(self, pid: int, class_name: str) -> list:
        """Visible top-level windows of that class owned by the process."""
        found = []

        def visit(hwnd, _):
            owner = self.wintypes.DWORD()
            self.user32.GetWindowThreadProcessId(hwnd, self.ctypes.byref(owner))
            if owner.value == pid and self.user32.IsWindowVisible(hwnd) and self.class_name(hwnd) == class_name:
                found.append(hwnd)
            return True

        self.user32.EnumWindows(self.callback(visit), 0)
        return found

    def controls(self, hwnd) -> dict:
        """Child windows of a dialog by control id."""
        found = {}

        def visit(child, _):
            found.setdefault(self.user32.GetDlgCtrlID(child), child)
            return True

        self.user32.EnumChildWindows(hwnd, self.callback(visit), 0)
        return found

    def choose(self, combo, text: str) -> bool:
        buffer = self.ctypes.create_unicode_buffer(text)
        index = self.user32.SendMessageW(combo, self.CB_FINDSTRINGEXACT, self.wintypes.WPARAM(-1 & 0xFFFFFFFFFFFFFFFF),
                                         self.ctypes.cast(buffer, self.ctypes.c_void_p).value)
        if index < 0:
            return False
        self.user32.SendMessageW(combo, self.CB_SETCURSEL, index, 0)
        self.user32.SendMessageW(self.user32.GetParent(combo), self.WM_COMMAND,
                                 (self.CBN_SELCHANGE << 16) | self.user32.GetDlgCtrlID(combo), combo)
        return True

    def client_area(self, hwnd) -> tuple[int, int, int, int]:
        """Screen position of the client area's corner and its size."""
        rect, corner = self.wintypes.RECT(), self.wintypes.POINT(0, 0)
        self.user32.GetClientRect(hwnd, self.ctypes.byref(rect))
        self.user32.ClientToScreen(hwnd, self.ctypes.byref(corner))
        return corner.x, corner.y, rect.right, rect.bottom

    def restore(self, hwnd, timeout_s: float = 5.) -> tuple[int, int, int, int]:
        """The client area, after bringing the window back if someone minimised it (a minimised window has none
        and its menu cannot be clicked)."""
        area = self.client_area(hwnd)
        if area[2] > 0 and area[3] > 0 and not self.user32.IsIconic(hwnd):
            return area
        self.user32.ShowWindow(hwnd, self.SW_RESTORE)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(.2)
            area = self.client_area(hwnd)
            if area[2] > 0 and area[3] > 0 and not self.user32.IsIconic(hwnd):
                time.sleep(1.)  # the game redraws its menu at the restored size
                return area
        raise LauncherError("the simulator window stays minimised and cannot be clicked")

    def click(self, hwnd, fraction: tuple[float, float]) -> None:
        left, top, width, height = self.restore(hwnd)
        x, y = int(width * fraction[0]), int(height * fraction[1])
        position = (y << 16) | (x & 0xFFFF)
        before = self.wintypes.POINT()
        self.user32.GetCursorPos(self.ctypes.byref(before))
        self.user32.SetForegroundWindow(hwnd)  # best effort; the posted click does not depend on it
        self.user32.SetCursorPos(left + x, top + y)
        try:
            time.sleep(.2)
            self.user32.PostMessageW(hwnd, self.WM_MOUSEMOVE, 0, position)
            time.sleep(.1)
            self.user32.PostMessageW(hwnd, self.WM_LBUTTONDOWN, self.MK_LBUTTON, position)
            time.sleep(.1)
            self.user32.PostMessageW(hwnd, self.WM_LBUTTONUP, 0, position)
            time.sleep(.25)
        finally:
            self.user32.SetCursorPos(before.x, before.y)

    def press(self, hwnd, key: int) -> None:
        scan = self.user32.MapVirtualKeyW(key, 0)
        self.user32.PostMessageW(hwnd, self.WM_KEYDOWN, key, 1 | (scan << 16))
        time.sleep(.1)
        self.user32.PostMessageW(hwnd, self.WM_KEYUP, key, 1 | (scan << 16) | (1 << 30) | (1 << 31))


class UdacityLauncher:
    def __init__(self, executable=None, *, directory=None, track: str = "lake", menu_wait_s: float = 4.):
        if track not in TRACK_BUTTONS:
            raise LauncherError(f"unknown track: {track}")
        self.executable = Path(executable) if executable else default_executable()
        self.directory = Path(directory) if directory else None
        self.track, self.menu_wait_s = track, float(menu_wait_s)
        self.process = None
        self.window = None
        self.clicks = 0
        self._win = None

    # ---- lifecycle -----------------------------------------------------------------------------
    def start(self) -> "UdacityLauncher":
        if sys.platform != "win32":
            raise LauncherError("the simulator can only be started automatically on Windows: start it yourself, "
                                "choose Autonomous Mode and pass --sim remote")
        binary = self.executable.resolve()
        if not binary.is_file():
            raise LauncherError(f"simulator not found: {binary} (set {ENVIRONMENT_VARIABLE} or unpack it there)")
        import psutil

        self._win = _Windows()
        self.process = subprocess.Popen([str(binary)], cwd=str(binary.parent))
        if self.directory is not None:
            # The supervisor of a guarded worker kills the simulator from this record if the worker dies.
            created = psutil.Process(self.process.pid).create_time()
            (self.directory / ".owned-simulator.json").write_text(
                json.dumps({"pid": self.process.pid, "create_time": created}), encoding="utf-8")
        self._configure()
        self.window = self._wait(lambda: self._win.windows(self.process.pid, "UnityWndClass"), 60.,
                                 "the simulator window did not appear")[0]
        time.sleep(self.menu_wait_s)  # the menu scene loads; nothing on the window says when it is ready
        self.enter()
        return self

    def _wait(self, probe, timeout_s: float, message: str):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise LauncherError(f"the simulator exited with code {self.process.returncode}")
            found = probe()
            if found:
                return found
            time.sleep(.2)
        raise LauncherError(message)

    def _configure(self) -> None:
        """Unity's start dialog: a small window, the lowest quality (no shadows on the road), then Play."""
        win = self._win

        def dialog_or_game():
            return win.windows(self.process.pid, "#32770") or win.windows(self.process.pid, "UnityWndClass")

        first = self._wait(dialog_or_game, 30., "the simulator opened no window")[0]
        if win.class_name(first) != "#32770":
            return  # a build without the dialog
        controls = win.controls(first)
        missing = [name for name, control in (("resolution", DIALOG_RESOLUTION), ("quality", DIALOG_QUALITY),
                                              ("Play", DIALOG_PLAY)) if control not in controls]
        if missing:
            raise LauncherError(f"unexpected configuration dialog: no {', '.join(missing)} control")
        if not win.choose(controls[DIALOG_RESOLUTION], RESOLUTION) or not win.choose(controls[DIALOG_QUALITY], QUALITY):
            raise LauncherError(f"the configuration dialog does not offer {RESOLUTION} / {QUALITY}")
        windowed = controls.get(DIALOG_WINDOWED)
        if windowed is not None and not win.user32.SendMessageW(windowed, win.BM_GETCHECK, 0, 0):
            win.user32.SendMessageW(windowed, win.BM_CLICK, 0, 0)
        win.user32.PostMessageW(controls[DIALOG_PLAY], win.BM_CLICK, 0, 0)

    def enter(self) -> None:
        """From the menu: choose the track, then Autonomous Mode."""
        if self.track != "lake":  # the lake track is selected when the menu opens
            self._win.click(self.window, TRACK_BUTTONS[self.track])
            time.sleep(.4)
        self._win.click(self.window, MODE_BUTTON)
        self.clicks += 1

    def restart(self) -> None:
        """Back to the menu and into the scene again: the car starts from the start line."""
        if self.process is None or self.process.poll() is not None:
            raise LauncherError("the simulator is not running")
        self._win.restore(self.window)  # a minimised game may not take the key
        self._win.press(self.window, self._win.VK_ESCAPE)
        time.sleep(2.)
        self.enter()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            from apps.console.jobs import kill_tree

            kill_tree(self.process.pid)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if self.directory is not None:
            (self.directory / ".owned-simulator.json").unlink(missing_ok=True)

    def describe(self) -> dict:
        return {"executable": self.executable.name, "track": self.track, "resolution": RESOLUTION, "quality": QUALITY,
                "mode_clicks": self.clicks}
