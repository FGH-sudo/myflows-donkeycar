"""Controllers and disturbances shared by the collectors.

The label of a frame is what the expert would do in the state the frame shows. The steering that is actually
applied may differ: ``Perturbation`` shifts it for a few frames at random times, so the car leaves its line and the
set also holds the views of drifting away and coming back, each still labelled with the expert's correction.
A set recorded from a perfect drive has no such views, and a model trained on it cannot recover from its own
small errors.
"""

from __future__ import annotations

import math


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clip(value: float, low: float = -1., high: float = 1.) -> float:
    if not all(math.isfinite(float(v)) for v in (value, low, high)) or low > high:
        raise ValueError("control values and bounds must be finite and ordered")
    return max(low, min(high, float(value)))


def finite_float(value) -> float:
    result = finite(value)
    if result is None:
        raise ValueError("value must be finite")
    return result


class CteExpert:
    """Steer back to the lane centre from a signed lateral error (PD on the error and its filtered rate).

    ``sign`` is the direction of the correction: -1 when a positive steering makes the error grow, which is how
    DonkeySim reports its cross-track error.
    """

    def __init__(self, kp: float = .3, kd: float = .25, *, sign: float = -1., rate_smoothing: float = .5):
        if not all(math.isfinite(float(v)) for v in (kp, kd, sign, rate_smoothing)) \
                or kp <= 0 or kd < 0 or sign == 0 or not 0 < rate_smoothing <= 1:
            raise ValueError("kp must be positive, kd non-negative and rate_smoothing in (0, 1]")
        self.kp, self.kd, self.sign, self.alpha = float(kp), float(kd), float(sign), float(rate_smoothing)
        self.reset()

    def reset(self) -> None:
        self.previous = None
        self.rate = 0.

    def __call__(self, error: float, now: float) -> float:
        if not math.isfinite(error) or not math.isfinite(now):
            raise ValueError("error and clock must be finite")
        if self.previous is not None:
            last, at = self.previous
            dt = now - at
            if dt > 1e-3:
                self.rate += self.alpha * ((error - last) / dt - self.rate)
        self.previous = (float(error), float(now))
        return clip(self.sign * (self.kp * error + self.kd * self.rate))


class SpeedHold:
    """Throttle that holds a target speed (PI); for simulators that take a throttle but report a speed."""

    def __init__(self, target: float, *, kp: float = .15, ki: float = .05, limit: float = 1.):
        if not all(math.isfinite(float(v)) for v in (target, kp, ki, limit)) \
                or target <= 0 or kp < 0 or ki < 0 or not 0 < limit <= 1:
            raise ValueError("target speed and limits must be finite and positive, gains non-negative")
        self.target, self.kp, self.ki, self.limit = float(target), float(kp), float(ki), float(limit)
        self.reset()

    def reset(self) -> None:
        self.integral = 0.
        self.at = None

    def __call__(self, speed: float, now: float) -> float:
        if not math.isfinite(speed) or not math.isfinite(now):
            raise ValueError("speed and clock must be finite")
        error = self.target - float(speed)
        if self.at is not None:
            # Bounded integral: a long standstill must not wind up into a launch.
            self.integral = clip(self.integral + error * max(0., min(.5, now - self.at)), -20., 20.)
        self.at = float(now)
        return clip(self.kp * error + self.ki * self.integral, -self.limit, self.limit)


class Perturbation:
    """Steering offsets of random sign, size and length at random times; zero in between."""

    def __init__(self, rng, *, interval_s=(2., 5.), frames=(3, 8), magnitude=(.3, .8), enabled: bool = True):
        self.rng = rng
        self.interval_s, self.frames, self.magnitude = tuple(interval_s), tuple(frames), tuple(magnitude)
        if any(len(values) != 2 or not all(math.isfinite(float(v)) for v in values)
               for values in (self.interval_s, self.frames, self.magnitude)) \
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in self.frames) \
                or not (0 < self.interval_s[0] <= self.interval_s[1] and 1 <= self.frames[0] <= self.frames[1]
                and 0 <= self.magnitude[0] <= self.magnitude[1]):
            raise ValueError("perturbation ranges must be ordered and positive")
        self.enabled = bool(enabled)
        self.pulses = 0
        self.reset(0.)

    def reset(self, now: float) -> None:
        self.left = 0
        self.value = 0.
        self.next_at = now + self.rng.uniform(*self.interval_s)

    def hold(self, now: float) -> None:
        """No disturbance now: a pulse under way ends, and the next one waits at least the shortest interval."""
        self.left = 0
        self.next_at = max(self.next_at, now + self.interval_s[0])

    def offset(self, now: float) -> float:
        """The offset for the frame at ``now``; call once per frame."""
        if not self.enabled:
            return 0.
        if self.left > 0:
            self.left -= 1
            if self.left == 0:
                self.next_at = now + self.rng.uniform(*self.interval_s)
            return self.value
        if now >= self.next_at:
            self.left = self.rng.randint(*self.frames) - 1
            self.value = self.rng.choice((-1., 1.)) * self.rng.uniform(*self.magnitude)
            self.pulses += 1
            if self.left == 0:
                self.next_at = now + self.rng.uniform(*self.interval_s)
            return self.value
        return 0.
