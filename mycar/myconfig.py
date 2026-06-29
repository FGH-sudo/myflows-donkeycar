# -*- coding: utf-8 -*-
"""Local DonkeyCar overrides for Windows simulator and MyFlows pilot."""

import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Windows + DonkeySimWin. Use "remote" if the simulator is already running.
DONKEY_GYM = True
DONKEY_SIM_PATH = os.path.join(REPO_ROOT, "DonkeySimWin", "donkey_sim.exe")
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0"
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
