"""What differs between simulators in a closed-loop run.

DonkeySim reports position, cross-track error and collisions, and the evaluation drives it at a fixed throttle.
The Udacity simulator reports the camera and the speed only: the run holds a target speed instead, measures the
distance from the speed, and takes the car's position across the road from the picture
(``apps.collect.vision``). Those picture measurements are recorded as estimates (``cte_source``), never as if
the simulator had reported them.

An environment opens the simulator session, supplies the per-episode ``monitor`` (None for DonkeySim) and
describes itself for the manifest and the report. The episode loop and the metrics are shared.
"""

from __future__ import annotations

from contextlib import contextmanager
import time

DONKEY = "donkey-generated-roads-v0"
UDACITY_FAKE = "udacity-fake-stadium"
UDACITY_TRACKS = ("lake", "jungle")
MPH = 0.44704  # metres per second in one mile per hour
FAILURES = ("collision", "stuck", "road_lost")


def _speed(info: dict) -> float:
    from apps.collect.policy import finite

    value = finite(info.get("speed"))
    if value is None:
        raise ValueError("simulator speed is missing or nonfinite")
    return value


class DonkeyEnvironment:
    title = "DonkeyCar 配对闭环评估"
    failure_label = "碰撞次数"
    cte_source = "telemetry"
    distance_unit = "simulator units"
    episode_overhead_s = 15.
    clock = staticmethod(time.perf_counter)

    def __init__(self, protocol: dict):
        from .simulator import FIXED_THROTTLE

        self.name = protocol.get("environment") or DONKEY
        self.fixed_throttle = FIXED_THROTTLE

    def session(self, out, *, executable=None, port=None):
        from .simulator import session

        return session(out, executable=executable, port=port)

    def monitor(self):
        return None

    def describe(self) -> dict:
        return {"fixed_throttle": self.fixed_throttle, "target_speed_mph": None, "cte_source": self.cte_source,
                "distance_unit": self.distance_unit}

    def intro(self) -> str:
        return (f"每个场景使用相同的默认 reset、客户端种子、终止规则、时限和 {self.fixed_throttle} 油门。"
                "模型顺序按场景轮换。位置距离使用连续有效 3D 遥测点之间的欧氏距离。")

    def footnote(self) -> str:
        return "Unity 场景随机种子和任意起点坐标无法通过当前接口指定。"


class UdacityMonitor:
    """Speed control and the camera-based stand-ins for what the Udacity link does not carry."""

    def __init__(self, *, speed_mph: float, stuck_s: float = 4., lost_frames: int = 30, warmup_s: float = 3.,
                 finder=None):
        from apps.collect.vision import RoadFinder

        self.default_speed = float(speed_mph)
        self.stuck_s, self.lost_frames, self.warmup_s = float(stuck_s), int(lost_frames), float(warmup_s)
        self.finder = finder or RoadFinder()
        self.reset({}, 0.)

    def reset(self, scenario: dict, now: float) -> None:
        from apps.collect.policy import SpeedHold

        self.hold = SpeedHold(float(scenario.get("speed_mph") or self.default_speed))
        self.finder.reset()
        self.start = self.previous = float(now)
        self.odometer = 0.
        self.lost = 0
        self.slow_since = None

    def throttle(self, info: dict, now: float) -> float:
        return self.hold(_speed(info), now)

    def inspect(self, image, info: dict, now: float) -> dict:
        speed = _speed(info)
        self.odometer += abs(speed) * MPH * max(0., now - self.previous)
        self.previous = now
        stop = None
        if now - self.start > self.warmup_s and speed < .5:
            self.slow_since = now if self.slow_since is None else self.slow_since
            if now - self.slow_since >= self.stuck_s:
                stop = "stuck"
        else:
            self.slow_since = None
        calibrated = self.finder.ready
        view = self.finder(image)
        self.lost = 0 if view.found else self.lost + 1
        if stop is None and self.lost >= self.lost_frames:
            stop = "road_lost"
        return {"cte": view.lateral if view.found and calibrated else None, "odometer": self.odometer, "stop": stop}


class UdacityEnvironment:
    failure_label = "失败次数"
    cte_source = "camera_estimate"
    distance_unit = "m"
    fixed_throttle = None
    episode_overhead_s = 45.  # a scene restart through the menu

    def __init__(self, protocol: dict):
        self.name = protocol["environment"]
        self.fake = self.name == UDACITY_FAKE
        self.track = self.name.split("-", 1)[1] if not self.fake else "stadium"
        if not self.fake and self.track not in UDACITY_TRACKS:
            raise ValueError(f"未知的 Udacity 赛道：{self.name}")
        self.title = "Udacity 模拟器配对闭环评估"
        self.speed_mph = float(protocol.get("target_speed_mph") or 12.)
        rules = protocol.get("rules") or {}
        self.stuck_s, self.lost_frames = float(rules.get("stuck_s", 4.)), int(rules.get("lost_frames", 30))
        self.clock = time.perf_counter

    @contextmanager
    def session(self, out, *, executable=None, port=None):
        from apps.sim.udacity import PORT, UdacityServer, UdacityStream, session

        if not self.fake:
            with session(out, executable=executable, port=port or PORT, track=self.track) as opened:
                yield opened
            return
        from apps.sim.udacity_fake import FakeUdacitySim

        with UdacityServer(port=0) as server, FakeUdacitySim(server.port) as fake:
            self.clock = lambda: fake.elapsed  # the stand-in runs on its own fixed time step
            yield (UdacityStream(server, timeout_s=10., restart=fake.restart, connect_timeout_s=20.),
                   {"environment": self.name, "simulator": "udacity-fake", "port": server.port, "owned_simulator": True,
                    "start_position_control": "scene restart"})

    def monitor(self):
        return UdacityMonitor(speed_mph=self.speed_mph, stuck_s=self.stuck_s, lost_frames=self.lost_frames)

    def describe(self) -> dict:
        return {"fixed_throttle": None, "target_speed_mph": self.speed_mph, "cte_source": self.cte_source,
                "distance_unit": self.distance_unit}

    def intro(self) -> str:
        return (f"每个场景从场景重启后的起点出发，保持各自的目标速度（默认 {self.speed_mph:g} mph），使用相同的终止规则和时限。"
                "模型顺序按场景轮换。模拟器不回传位置、CTE 和碰撞：距离由速度积分得到（米）；"
                "表中的 CTE 是由前视图像估计的横向位置（0 在起步时确定的行驶线上，+1 为右边缘，−1 为向左同样远），不是模拟器真值；"
                "失败指车速持续为零（卡住）或图像中连续找不到路面（丢路）。")

    def footnote(self) -> str:
        return "模拟器没有重置接口，每个片段都通过菜单重启场景；起点固定，种子只用于记录。"


def for_protocol(protocol: dict):
    name = protocol.get("environment") or DONKEY
    if isinstance(name, str) and name.startswith("udacity-"):
        return UdacityEnvironment(protocol)
    return DonkeyEnvironment(protocol)
