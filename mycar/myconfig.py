# -*- coding: utf-8 -*-
"""Local DonkeyCar overrides for the simulator and MyFlows pilot."""

import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Simulator unpacked into the repository root from the gym-donkeycar release for this platform
# (DonkeySimWin, DonkeySimLinux or DonkeySimMac). The DONKEY_SIM_PATH environment variable
# overrides it; "remote" uses a simulator that is already running.
DONKEY_GYM = True
_simulator = {
    "win32": ("DonkeySimWin", "donkey_sim.exe"),
    "darwin": ("DonkeySimMac", "donkey_sim.app", "Contents", "MacOS", "donkey_sim"),
}.get(sys.platform, ("DonkeySimLinux", "donkey_sim.x86_64"))
DONKEY_SIM_PATH = os.environ.get("DONKEY_SIM_PATH") or os.path.join(REPO_ROOT, *_simulator)
# gym-donkeycar: donkey-generated-roads-v0 -> generated_road（与 generated-road 训练数据一致）
# donkey-generated-track-v0 -> generated_track（环形赛道，训练数据未覆盖）
DONKEY_GYM_ENV_NAME = "donkey-generated-roads-v0"
SIM_HOST = "127.0.0.1"

# Web control defaults for data collection in the simulator.
WEB_INIT_MODE = "user"
WEB_DEFAULT_THROTTLE = 0.0
WEB_MAX_THROTTLE = 0.4
WEB_BUTTON_STEERING_ENABLE = True
USE_JOYSTICK_AS_DEFAULT = False
CONTROLLER_TYPE = "mock"

# MyFlows fixed-speed steering pilot. Keep this aligned with training
# --fixed-throttle unless intentionally changing simulator speed.
MYFLOWS_FIXED_THROTTLE = 0.2
MYFLOWS_MAX_THROTTLE = 0.2
MYFLOWS_STEERING_SCALE = 1.0
MYFLOWS_DEBUG = False
MYFLOWS_DEVICE = "cuda"  # auto | cuda | cpu
