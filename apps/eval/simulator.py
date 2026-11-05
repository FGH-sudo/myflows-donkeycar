"""Bounded local simulator sessions. Expose only telemetry observed on the wire."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXED_THROTTLE = 0.2


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def guarded_worker(module, argv, timeout_s, directory):
    """Also bound SDK startup/reset calls that may wait indefinitely for a Unity frame."""
    from apps.console.jobs import kill_tree

    command = [sys.executable, "-X", "utf8", "-m", module, *argv, "--worker"]
    with (Path(directory) / "session.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=timeout_s)
        except BaseException:
            kill_tree(proc.pid)
            proc.wait(timeout=10)
            raise
        finally:
            # A hard crash of the worker can re-parent Unity before the supervisor sees it.
            owner = Path(directory) / ".owned-simulator.json"
            if owner.is_file():
                import psutil

                recorded = json.loads(owner.read_text(encoding="utf-8"))
                try:
                    unity = psutil.Process(recorded["pid"])
                    if abs(unity.create_time() - recorded["create_time"]) < .001:
                        kill_tree(unity.pid)
                except psutil.NoSuchProcess:
                    pass
                owner.unlink(missing_ok=True)
    if code:
        raise RuntimeError(f"simulator worker exited {code}, see {directory}/session.log")


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FrameStream:
    """Advance on a completed camera telemetry message, with a finite receive deadline."""

    def __init__(self, env, timeout_s):
        self.env = env
        self.viewer = env.unwrapped.viewer
        self.timeout_s = timeout_s
        self.condition = threading.Condition()
        self.seq = 0
        self.latest = None
        handler = self.viewer.handler
        self._telemetry = handler.fns["telemetry"]
        self._line_crossing = handler.fns["collision_with_starting_line"]
        self.line_events = 0
        handler.fns["telemetry"] = self._capture
        handler.fns["collision_with_starting_line"] = self._crossing
        # Apply the evaluation's own locked collision/CTE rules to the raw hit and CTE fields.
        env.unwrapped.set_episode_over_fn(lambda handler: None)

    def _crossing(self, message):
        self.line_events += 1
        self._line_crossing(message)

    def _capture(self, message):
        self._telemetry(message)
        handler = self.viewer.handler
        info = {key: message.get(key) for key in ("cte", "speed", "hit")}
        for name, components in {"pos": ("pos_x", "pos_y", "pos_z"), "gyro": ("gyro_x", "gyro_y", "gyro_z"),
                                 "accel": ("accel_x", "accel_y", "accel_z"), "vel": ("vel_x", "vel_y", "vel_z"),
                                 "car": ("roll", "pitch", "yaw")}.items():
            info[name] = [message[key] for key in components] if all(key in message for key in components) else None
        info.update(lidar_points=len(handler.lidar) if "lidar" in message else None,
                    forward_vel=handler.forward_vel if info["vel"] is not None and info["car"] is not None else None,
                    active_node=message.get("activeNode"), total_nodes=message.get("totalNodes"),
                    simulation_time=message.get("time"), observed_steering=message.get("steering_angle"),
                    observed_throttle=message.get("throttle"),
                    lap_count=handler.lap_count, last_lap_time=handler.last_lap_time, line_events=self.line_events,
                    wire_fields=sorted(message), monotonic_s=time.perf_counter(), unix_s=time.time())
        with self.condition:
            self.seq += 1
            self.latest = (handler.image_array, dict(info, frame_id=self.seq))
            self.condition.notify_all()

    def _next(self, previous):
        with self.condition:
            if not self.condition.wait_for(lambda: self.seq > previous, self.timeout_s):
                raise TimeoutError(f"no camera telemetry for {self.timeout_s:g} seconds")
            return self.latest

    def reset(self, *, seed):
        self.line_events = 0
        self.env.reset(seed=seed)
        # SDK reset clears cached fields. Use a new, complete frame after reset returns.
        return self._next(self.seq)

    def step(self, action):
        with self.condition:
            previous = self.seq
            self.viewer.take_action(np.asarray(action, dtype=np.float32))
        return self._next(previous)

    def brake(self):
        self.viewer.handler.send_control(0., 0., 1.)

    def close(self):
        self.brake()
        handler = self.viewer.handler
        handler.fns["telemetry"] = self._telemetry
        handler.fns["collision_with_starting_line"] = self._line_crossing


@contextmanager
def session(directory, *, executable=None, port=None, frame_timeout_s=5.):
    """Own the launched Unity process, or connect to a caller-owned simulator with 'remote'."""
    import gymnasium as gym
    import gym_donkeycar  # registers the Gymnasium environments
    import psutil
    from apps.console.jobs import kill_tree
    from mycar.myconfig import DONKEY_GYM_ENV_NAME, DONKEY_SIM_PATH

    executable = executable or DONKEY_SIM_PATH
    remote = executable == "remote"
    port = port or (9091 if remote else _free_port())
    directory = Path(directory).resolve()
    proc = env = stream = None
    try:
        if not remote:
            binary = Path(executable).resolve(strict=True)
            command = [str(binary), "--host", "127.0.0.1", "--port", str(port), "-logFile", str(directory / "unity.log"),
                       "-screen-fullscreen", "0", "-screen-width", "640", "-screen-height", "480"]
            proc = subprocess.Popen(command, cwd=directory)
            unity = psutil.Process(proc.pid)
            write_json(directory / ".owned-simulator.json", {"pid": unity.pid, "create_time": unity.create_time()})
            deadline = time.monotonic() + 60
            ready_message = f"Server Listening on: 127.0.0.1:{port}"
            while proc.poll() is None:
                try:
                    unity_log = (directory / "unity.log").read_text(encoding="utf-8", errors="replace")
                except OSError:
                    unity_log = ""
                if ready_message in unity_log or any(c.status == psutil.CONN_LISTEN and c.laddr.port == port
                                                     for c in unity.net_connections(kind="tcp")):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Unity did not open its simulator port")
                time.sleep(.15)
            if proc.poll() is not None:
                raise RuntimeError(f"Unity exited during startup with code {proc.returncode}")
        env = gym.make(DONKEY_GYM_ENV_NAME, conf={"host": "127.0.0.1", "port": port, "log_level": logging.WARNING,
                                                "cam_resolution": (120, 160, 3), "frame_skip": 1},
                       disable_env_checker=True)
        stream = FrameStream(env, frame_timeout_s)
        yield stream, {"environment": DONKEY_GYM_ENV_NAME, "gym_donkeycar": gym_donkeycar.__version__,
                       "gymnasium": gym.__version__, "owned_simulator": not remote, "fixed_throttle": FIXED_THROTTLE,
                       "client_seed": True, "unity_seed_control": False, "start_position_control": "simulator default reset"}
    finally:
        try:
            if stream is not None:
                stream.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if proc is not None and proc.poll() is None:
                    kill_tree(proc.pid)
                    proc.wait(timeout=10)
                if proc is not None:
                    (directory / ".owned-simulator.json").unlink(missing_ok=True)
