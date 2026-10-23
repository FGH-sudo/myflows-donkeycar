"""Locate the Nsight Systems (nsys) and Nsight Compute (ncu) command-line tools on Windows and Linux.

Search order: an explicit path, then the MF_NSYS / MF_NCU environment variable, then PATH, then
the newest version in the installers' default locations.
"""

import glob
import os
import re
import shutil
from pathlib import Path

ENVIRONMENT = {"nsys": "MF_NSYS", "ncu": "MF_NCU"}
# Default install locations; "*" is the version directory.
if os.name == "nt":
    _INSTALLS = {
        "nsys": ("C:/Program Files/NVIDIA Corporation/Nsight Systems */target-windows-x64/nsys.exe",),
        "ncu": ("C:/Program Files/NVIDIA Corporation/Nsight Compute */target/windows-desktop-win7-x64/ncu.exe",),
    }
else:
    _INSTALLS = {
        "nsys": ("/opt/nvidia/nsight-systems/*/bin/nsys", "/usr/local/cuda/bin/nsys"),
        "ncu": ("/opt/nvidia/nsight-compute/*/ncu", "/usr/local/cuda/bin/ncu"),
    }


def _version(path):
    match = re.search(r"\d+(?:\.\d+)+", path)
    return tuple(int(part) for part in match.group().split(".")) if match else ()


def find_tool(name, explicit=None):
    """Path of the ``nsys`` or ``ncu`` executable."""
    chosen = explicit or os.environ.get(ENVIRONMENT[name])
    if chosen:
        if not Path(chosen).is_file():
            raise FileNotFoundError(f"{name} not found at {chosen}")
        return Path(chosen)
    found = shutil.which(name)
    if found and Path(found).suffix.lower() in (".bat", ".cmd"):
        # Windows puts the ncu.bat launcher on PATH. It runs through cmd.exe, which splits a kernel
        # regex at "|", so use the executable it starts.
        found = next(iter(sorted(glob.glob(str(Path(found).parent / "target" / "*" / f"{name}.exe")))), None)
    if found:
        return Path(found)
    for pattern in _INSTALLS[name]:
        matches = sorted(glob.glob(pattern), key=_version, reverse=True)
        if matches:
            return Path(matches[0])
    raise FileNotFoundError(f"{name} not found: pass its path, set {ENVIRONMENT[name]}, or put it on PATH")
