"""Talk to the Udacity self-driving-car simulator (term 1) the way its own ``drive.py`` does.

The simulator is a Socket.IO 1.x client. In autonomous mode it opens
``ws://127.0.0.1:4567/socket.io/?EIO=4&transport=websocket``, sends ``telemetry`` events (steering angle in
degrees, throttle, speed in mph and the centre camera as a base64 JPEG) and applies ``steer`` events. It sends
the next telemetry only after it received a ``steer`` or ``manual`` event, so both sides advance in lock step.

python-socketio 5 no longer speaks this protocol revision and the old releases are unmaintained. The part the
simulator uses is small — Engine.IO 3 framing over one WebSocket, text packets only — and is implemented here
on the ``websockets`` package. ``UdacityStream`` gives the link the interface of the DonkeySim session
(``reset`` / ``step`` / ``brake``), so collection and closed-loop evaluation drive both simulators with the
same code.

The wire carries no position, cross-track error or collision flag, and no reset message: an episode can only be
restarted through the simulator's menu (``apps.sim.udacity_launcher`` does that on Windows).
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import io
import json
import math
import secrets
import threading
import time

import numpy as np

HOST = "127.0.0.1"
PORT = 4567  # fixed in the simulator build
MAX_STEER_DEG = 25.  # the simulator's steering lock; telemetry reports degrees, steer events take [-1, 1]
RATE_HZ = 20.  # commands per second with the real simulator; DonkeySim's telemetry runs at the same rate
PING_INTERVAL_MS, PING_TIMEOUT_MS = 25000, 60000
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 1024 * 1024
# Engine.IO packet types and the Socket.IO types inside an Engine.IO message.
EIO_OPEN, EIO_CLOSE, EIO_PING, EIO_PONG, EIO_MESSAGE = "0", "1", "2", "3", "4"
SIO_CONNECT, SIO_DISCONNECT, SIO_EVENT = "0", "1", "2"


class SimulatorGone(ConnectionError):
    pass


def event_packet(name: str, data: dict) -> str:
    return EIO_MESSAGE + SIO_EVENT + json.dumps([name, data], separators=(",", ":"))


def parse_event(packet: str):
    """``(name, data)`` of a Socket.IO event packet, or None for anything else (connect, ack, malformed)."""
    if not packet.startswith(EIO_MESSAGE + SIO_EVENT):
        return None
    start = packet.find("[")  # an optional ack id sits between the type and the payload
    if start < 0 or not packet[2:start].isdigit() and packet[2:start] != "":
        return None
    try:
        payload = json.loads(packet[start:])
    except ValueError:
        return None
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], str):
        return None
    return payload[0], (payload[1] if len(payload) > 1 else None)


def number(text, default=None):
    """The simulator formats numbers with the machine's culture ("N4"): accept a decimal comma as well."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        try:
            value = float(str(text).replace(" ", "").replace(",", "."))
        except ValueError:
            return default
    return value if math.isfinite(value) else default


def decode_image(text: str) -> np.ndarray:
    import cv2
    from PIL import Image

    try:
        payload = base64.b64decode(text, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("camera image is not base64") from exc
    try:
        with Image.open(io.BytesIO(payload)) as header:
            width, height = header.size
            if header.format not in ("JPEG", "PNG") or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("camera image exceeds the pixel limit or has an unsupported format")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("camera image has an invalid header") from exc
    raw = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    if bgr is None:
        raise ValueError("camera image is not a decodable picture")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class UdacityServer:
    """The port the simulator connects to. Keeps the newest telemetry; one simulator at a time."""

    def __init__(self, host: str = HOST, port: int = PORT):
        self.host, self.port = host, int(port)
        self._condition = threading.Condition()
        self._connection = None
        self._server = None
        self._thread = None
        self.seq = 0
        self.connects = 0
        self.manual_frames = 0
        self._latest = None

    # ---- lifecycle ---------------------------------------------------------------------------
    def start(self) -> "UdacityServer":
        from websockets.sync.server import serve

        # No per-message compression and no WebSocket pings: the simulator's client library negotiates neither,
        # and Engine.IO has its own heartbeat (the client pings, the server answers).
        self._server = serve(self._serve, self.host, self.port, compression=None, ping_interval=None,
                             max_size=MAX_MESSAGE_BYTES, server_header=None, origins=[None])
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="udacity-socketio", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ---- the simulator's side ----------------------------------------------------------------
    def _serve(self, connection) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            with self._condition:
                if self._connection is not None:
                    busy = True
                else:
                    self._connection = connection
                    busy = False
            if busy:
                connection.close(1008, "a simulator is already connected")
                return
            connection.send(EIO_OPEN + json.dumps({"sid": secrets.token_hex(10), "upgrades": [],
                                                   "pingInterval": PING_INTERVAL_MS, "pingTimeout": PING_TIMEOUT_MS}))
            connection.send(EIO_MESSAGE + SIO_CONNECT)
            with self._condition:
                self.connects += 1
                self._condition.notify_all()
            for packet in connection:
                if not isinstance(packet, str) or not packet:
                    continue
                if packet[0] == EIO_PING:
                    connection.send(EIO_PONG + packet[1:])
                elif packet[0] == EIO_CLOSE or packet.startswith(EIO_MESSAGE + SIO_DISCONNECT):
                    break
                else:
                    self._event(connection, packet)
        except ConnectionClosed:
            pass
        finally:
            with self._condition:
                if self._connection is connection:
                    self._connection = None
                self._condition.notify_all()

    def _event(self, connection, packet: str) -> None:
        event = parse_event(packet)
        if event is None or event[0] != "telemetry":
            return
        data = event[1]
        if not isinstance(data, dict) or not data.get("image"):
            # The simulator is being driven by hand: it waits for this before it reports again.
            self.manual_frames += 1
            connection.send(event_packet("manual", {}))
            return
        with self._condition:
            if self._connection is not connection:
                return
            self.seq += 1
            self._latest = (self.seq, data, time.perf_counter(), time.time())
            self._condition.notify_all()

    # ---- the driver's side -------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connection is not None

    def wait_connected(self, timeout_s: float, *, after: int = 0) -> bool:
        """Wait for a connection newer than the ``after``-th one."""
        with self._condition:
            return self._condition.wait_for(lambda: self.connects > after and self._connection is not None, timeout_s)

    def next_frame(self, after: int, timeout_s: float):
        """The newest telemetry with a sequence number above ``after``: ``(seq, fields, monotonic_s, unix_s)``."""
        with self._condition:
            if not self._condition.wait_for(lambda: self.seq > after, timeout_s):
                if self._connection is None:
                    raise SimulatorGone("the simulator is not connected")
                raise TimeoutError(f"no telemetry for {timeout_s:g} seconds")
            return self._latest

    def send_control(self, steering: float, throttle: float) -> None:
        steering, throttle = float(steering), float(throttle)
        if not all(math.isfinite(v) and -1 <= v <= 1 for v in (steering, throttle)):
            raise ValueError("steering and throttle must be finite values in [-1, 1]")
        connection = self._connection
        if connection is None:
            raise SimulatorGone("the simulator is not connected")
        from websockets.exceptions import ConnectionClosed

        # The simulator reads both values with GetField(...).str: they must be JSON strings.
        packet = event_packet("steer", {"steering_angle": f"{float(steering):.4f}", "throttle": f"{float(throttle):.4f}"})
        try:
            connection.send(packet)
        except ConnectionClosed as exc:
            raise SimulatorGone("the simulator closed the connection") from exc


class UdacityStream:
    """Frames and controls in the shape of the DonkeySim session.

    ``restart`` (optional) puts the car back on the start line by restarting the scene; without it ``reset``
    continues from wherever the car is, which is all a simulator started by hand allows.
    """

    def __init__(self, server: UdacityServer, *, timeout_s: float = 10., restart=None, connect_timeout_s: float = 60.,
                 rate_hz: float | None = None):
        self.server, self.timeout_s, self.restart = server, float(timeout_s), restart
        self.connect_timeout_s = float(connect_timeout_s)
        # The simulator answers every command with a frame at once, a few hundred times a second on a fast
        # machine. ``rate_hz`` spaces the commands so that a second of driving is that many frames, not 250.
        self.period = 1. / rate_hz if rate_hz else 0.
        self._next_at = 0.
        self.resets = 0

    def _frame(self, after: int):
        seq, data, monotonic_s, unix_s = self.server.next_frame(after, self.timeout_s)
        image = decode_image(data["image"])
        steering = number(data.get("steering_angle"))
        info = {"frame_id": seq, "monotonic_s": monotonic_s, "unix_s": unix_s, "speed": number(data.get("speed")),
                "observed_steering": None if steering is None else steering / MAX_STEER_DEG,
                "observed_throttle": number(data.get("throttle")), "cte": None, "pos": None, "hit": None,
                "lap_count": None, "wire_fields": sorted(data)}
        return image, info

    def reset(self, *, seed=None):
        """A fresh frame of a (re)started episode. ``seed`` is accepted for interface parity; the simulator has
        no random state to seed."""
        before = self.server.connects
        if self.restart is not None and self.resets:
            self.restart()
            if not self.server.wait_connected(self.connect_timeout_s, after=before):
                raise SimulatorGone("the simulator did not reconnect after the restart")
        elif not self.server.wait_connected(self.connect_timeout_s):
            raise SimulatorGone("the simulator did not connect; start it and choose Autonomous Mode")
        self.resets += 1
        # A neutral command makes the simulator report its present view, whether or not it already sent one.
        previous = self.server.seq
        self.server.send_control(0., 0.)
        return self._frame(previous)

    def step(self, action):
        if self.period:
            now = time.perf_counter()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            self._next_at = max(self._next_at + self.period, time.perf_counter())
        previous = self.server.seq
        self.server.send_control(float(action[0]), float(action[1]))
        return self._frame(previous)

    def brake(self, *, limit_s: float = 3.):
        """Bring the car to a stop (a negative throttle brakes while it rolls forward); never raises."""
        deadline = time.monotonic() + limit_s
        try:
            while time.monotonic() < deadline:
                _image, info = self.step((0., -1.))
                if (info["speed"] or 0.) < .5:
                    break
            self.server.send_control(0., 0.)
        except (ConnectionError, TimeoutError, ValueError):
            pass

    def close(self):
        self.brake()


@contextmanager
def session(directory=None, *, executable=None, port: int = PORT, frame_timeout_s: float = 10., track: str = "lake",
            connect_timeout_s: float = 90., rate_hz: float = RATE_HZ):
    """Listen for the simulator and, unless ``executable`` is ``remote``, start it and enter autonomous mode."""
    server = UdacityServer(port=port).start()
    launcher = None
    try:
        restart = None
        if executable != "remote":
            from .udacity_launcher import UdacityLauncher

            launcher = UdacityLauncher(executable, directory=directory, track=track)

            def entered(after: int) -> None:
                """The scene has started when the simulator connects; a click that missed is repeated."""
                for _ in range(3):
                    if server.wait_connected(20., after=after):
                        return
                    launcher.enter()
                if not server.wait_connected(20., after=after):
                    raise SimulatorGone("the simulator did not enter autonomous mode")

            def restart() -> None:
                before = server.connects
                from .udacity_launcher import LauncherError

                try:
                    if launcher.process is None or launcher.process.poll() is not None:
                        raise LauncherError("the owned simulator exited")
                    launcher.restart()
                    entered(before)
                    return
                except (LauncherError, SimulatorGone):
                    # A dead process or an unresponsive menu is recoverable because this session owns it.
                    launcher.stop()
                    launcher.start()
                entered(before)

            launcher.start()
            entered(0)
        stream = UdacityStream(server, timeout_s=frame_timeout_s, restart=restart, connect_timeout_s=connect_timeout_s,
                               rate_hz=rate_hz)
        yield stream, {"environment": f"udacity-{track}", "simulator": "udacity-term1", "port": server.port,
                       "rate_hz": rate_hz,
                       "owned_simulator": launcher is not None, "max_steer_deg": MAX_STEER_DEG,
                       "launcher": launcher.describe() if launcher is not None else None,
                       "start_position_control": "scene restart" if launcher is not None else "none (remote simulator)"}
    finally:
        try:
            if server.connected:
                UdacityStream(server, timeout_s=2.).brake(limit_s=2.)
        finally:
            try:
                if launcher is not None:
                    launcher.stop()
            finally:
                server.close()
