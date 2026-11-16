"""A stand-in for the Udacity simulator that speaks its wire protocol.

A kinematic car on a stadium-shaped track (two straights joined by half circles), driven anticlockwise like the
lake track, with a rendered centre-camera view. Like the real program it is a Socket.IO 1.x *client*: it
connects to the driver's port, reports telemetry and waits for a ``steer`` event before it reports again. Time
advances by a fixed step per ``steer`` event, so a run does not depend on how fast the host is.

It exists so that the Udacity path — protocol, expert, collector, drive bridge and closed-loop evaluation — can
run without the Unity build: in the unit tests, in CI and for a dry run on a machine without the simulator::

    python -m apps.sim.udacity_fake      # then start a driver, for example python -m apps.sim.udacity_drive

Unlike the real simulator it knows where the car is: ``FakeUdacitySim.history`` holds the true lateral offset of
every frame, which is what the tests compare the camera-based estimates with. None of that is sent on the wire.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import time

import numpy as np

from .udacity import EIO_MESSAGE, EIO_OPEN, EIO_PING, HOST, MAX_STEER_DEG, PORT, event_packet, parse_event

MPH = 0.44704  # metres per second in one mile per hour
WIDTH, HEIGHT, HORIZON, FOCAL, CAMERA_HEIGHT = 320, 160, 73, 200., 1.5  # the real camera's vanishing row
HOOD_ROWS = 10  # the real simulator shows only a sliver of the hood along the bottom edge
ROAD, LINE, GROUND, HOOD = (108, 108, 110), (205, 185, 60), (92, 128, 74), (40, 42, 48)
FRAGMENT_BYTES = 1016  # the simulator's WebSocket library splits larger messages into fragments of this size


class StadiumTrack:
    """Centre line: two straights of length ``straight`` on y = -radius and y = +radius, joined by half circles."""

    def __init__(self, straight: float = 80., radius: float = 30., half_width: float = 4.):
        self.straight, self.radius, self.half_width = float(straight), float(radius), float(half_width)
        self.length = 2 * self.straight + 2 * math.pi * self.radius

    def offset(self, x, y):
        """Signed distance from the centre line, positive to the right of the anticlockwise driving direction."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        cx = np.clip(x, 0., self.straight)
        return np.hypot(x - cx, y) - self.radius

    def heading(self, x: float, y: float) -> float:
        """Driving direction of the centre line at the point nearest to (x, y)."""
        if 0. <= x <= self.straight:
            return 0. if y < 0 else math.pi
        centre = 0. if x < 0 else self.straight
        return math.atan2(y, x - centre) + math.pi / 2

    def start(self):
        return 5., -self.radius, 0.


class FakeCar:
    WHEELBASE = 2.7

    def __init__(self, track: StadiumTrack):
        self.track = track
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.heading = self.track.start()
        self.speed = 0.  # m/s
        self.steering = 0.  # [-1, 1], positive to the right
        self.throttle = 0.
        self.distance = 0.
        self.stuck = False

    @property
    def offset(self) -> float:
        return float(self.track.offset(self.x, self.y))

    def step(self, steering: float, throttle: float, dt: float) -> None:
        self.steering = max(-1., min(1., float(steering)))
        self.throttle = max(-1., min(1., float(throttle)))
        if abs(self.offset) > self.track.half_width + 2.:
            self.stuck = True  # off the road for good, like a car on the kerb stones or in the lake
        if self.stuck:
            self.speed = 0.
            return
        if self.throttle >= 0:
            acceleration = 5. * self.throttle - .37 * self.speed
        else:
            acceleration = 12. * self.throttle - .37 * self.speed
        self.speed = max(0., self.speed + acceleration * dt)
        angle = math.radians(self.steering * MAX_STEER_DEG)
        self.heading -= self.speed * math.tan(angle) / self.WHEELBASE * dt
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        self.distance += self.speed * dt


def render(track: StadiumTrack, car: FakeCar, rng=None) -> np.ndarray:
    """The centre camera: a pinhole view of the ground plane with the road, its edge lines and the car's hood."""
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
    sky = np.linspace(0., 1., HORIZON, dtype=np.float32)[:, None, None]
    image[:HORIZON] = (1 - sky) * np.array((95, 140, 215), np.float32) + sky * np.array((170, 200, 235), np.float32)
    rows = np.arange(HORIZON, HEIGHT, dtype=np.float64)[:, None] + .5
    cols = np.arange(WIDTH, dtype=np.float64)[None, :] + .5
    forward = FOCAL * CAMERA_HEIGHT / np.maximum(rows - HORIZON, 1e-6)
    right = (cols - WIDTH / 2) * forward / FOCAL
    cos, sin = math.cos(car.heading), math.sin(car.heading)
    offset = np.abs(track.offset(car.x + forward * cos + right * sin, car.y + forward * sin - right * cos))
    ground = image[HORIZON:]
    ground[:] = GROUND
    ground[offset < track.half_width] = ROAD
    ground[(offset < track.half_width) & (offset > track.half_width - .35)] = LINE
    if rng is not None:
        image += rng.normal(0., 3., size=image.shape).astype(np.float32)
    for row in range(HEIGHT - HOOD_ROWS, HEIGHT):  # hood: a dark trapezoid along the bottom edge
        inset = int((HEIGHT - 1 - row) * 6)
        image[row, 60 + inset:WIDTH - 60 - inset] = HOOD
    return np.clip(image, 0, 255).astype(np.uint8)


class FakeUdacitySim:
    """Runs the car and the protocol client in a background thread."""

    def __init__(self, port: int = PORT, host: str = HOST, *, dt: float = .05, track: StadiumTrack | None = None,
                 seed: int | None = 0, fragment_bytes: int = FRAGMENT_BYTES, ping_every: int = 0, jpeg_quality: int = 90):
        self.url = f"ws://{host}:{int(port)}/socket.io/?EIO=4&transport=websocket"
        self.dt = float(dt)
        self.track = track or StadiumTrack()
        self.car = FakeCar(self.track)
        self.rng = None if seed is None else np.random.default_rng(seed)
        self.fragment_bytes, self.ping_every, self.jpeg_quality = int(fragment_bytes), int(ping_every), int(jpeg_quality)
        self.history: list[dict] = []
        self.frames = self.pongs = self.connections = self.restarts = 0
        self.elapsed = 0.
        self.error = None
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._connection = None
        self._thread = None

    # ---- control from the test or the command line ------------------------------------------
    def start(self) -> "FakeUdacitySim":
        self._thread = threading.Thread(target=self._run, name="fake-udacity-sim", daemon=True)
        self._thread.start()
        return self

    def restart(self) -> None:
        """What choosing the mode again in the menu does: the scene reloads and the client connects anew."""
        self.restarts += 1
        self._restart.set()
        self._drop()

    def stop(self) -> None:
        self._stop.set()
        self._drop()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _drop(self) -> None:
        connection = self._connection
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- the simulator --------------------------------------------------------------------------
    def _telemetry(self) -> str:
        import cv2

        frame = render(self.track, self.car, self.rng)
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        self.frames += 1
        self.history.append({"frame": self.frames, "t": self.elapsed, "x": self.car.x, "y": self.car.y,
                             "offset": self.car.offset, "speed_mph": self.car.speed / MPH,
                             "steering": self.car.steering, "throttle": self.car.throttle, "stuck": self.car.stuck,
                             "distance": self.car.distance})
        # The simulator formats every value as text with four decimals.
        return event_packet("telemetry", {"steering_angle": f"{self.car.steering * MAX_STEER_DEG:.4f}",
                                          "throttle": f"{self.car.throttle:.4f}", "speed": f"{self.car.speed / MPH:.4f}",
                                          "image": base64.b64encode(encoded.tobytes()).decode("ascii")})

    def _send(self, connection, packet: str) -> None:
        if self.fragment_bytes > 0 and len(packet) > self.fragment_bytes:
            connection.send([packet[i:i + self.fragment_bytes] for i in range(0, len(packet), self.fragment_bytes)])
        else:
            connection.send(packet)

    def _serve(self, connection) -> None:
        self._send(connection, self._telemetry())  # the socket opened
        for packet in connection:
            if self._stop.is_set() or self._restart.is_set():
                return
            if not isinstance(packet, str) or not packet:
                continue
            if packet[0] == EIO_OPEN:
                self._send(connection, self._telemetry())  # the Engine.IO handshake raises "open" a second time
                continue
            if packet[0] == "3":
                self.pongs += 1
                continue
            event = parse_event(packet) if packet[0] == EIO_MESSAGE else None
            if event is None:
                continue
            name, data = event
            if name == "steer":
                self.car.step(float(data["steering_angle"]), float(data["throttle"]), self.dt)
                self.elapsed += self.dt
            elif name != "manual":
                continue
            if self.ping_every and self.frames % self.ping_every == 0:
                connection.send(EIO_PING)
            self._send(connection, self._telemetry())

    def _run(self) -> None:
        from websockets.exceptions import ConnectionClosed, InvalidHandshake
        from websockets.sync.client import connect

        while not self._stop.is_set():
            if self._restart.is_set():
                self._restart.clear()
                self.car.reset()
            try:
                with connect(self.url, compression=None, ping_interval=None, max_size=None, open_timeout=5) as connection:
                    self._connection = connection
                    self.connections += 1
                    self._serve(connection)
            except (OSError, InvalidHandshake, TimeoutError):
                time.sleep(.1)  # the driver is not listening yet: keep trying, as the simulator does
            except ConnectionClosed:
                pass
            except Exception as exc:  # a bug in the stand-in must fail the test, not hang it
                self.error = exc
                return
            finally:
                self._connection = None
            if not self._restart.is_set() and not self._stop.is_set():
                time.sleep(.05)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--seconds", type=float, default=0., help="stop after this long (0 = until interrupted)")
    args = parser.parse_args(argv)
    sim = FakeUdacitySim(args.port).start()
    began = time.monotonic()
    try:
        while not args.seconds or time.monotonic() - began < args.seconds:
            time.sleep(1.)
            if sim.error is not None:
                raise sim.error
            last = sim.history[-1] if sim.history else {}
            print(json.dumps({"frames": sim.frames, "offset": round(last.get("offset", 0.), 3),
                              "speed_mph": round(last.get("speed_mph", 0.), 2)}), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
