"""Find the road edges in a forward camera picture and steer along them.

The Udacity simulator reports no position, so the expert of its collector has to see. The surface right in
front of the car is taken as the road colour. On every scan line the road ends, to either side, at the first
thing that is clearly not that colour: a lane line, a kerb, the verge, a wall. Lightness is nearly ignored —
asphalt is speckled and gets paler with distance — and the colour itself decides.

The edge points of all scan lines are then explained by one model of a road on flat ground. A point ``d`` rows
below the vanishing row lies at a distance proportional to ``1 / d``, so an edge at lateral offset ``Y`` of a
road with heading error ``psi`` and curvature ``kappa`` appears at column

    x(d) = c0 + c1 * d + c2 / d        c0: vanishing point (heading), c1 ~ Y, c2 ~ kappa

with the same ``c0`` and ``c2`` for both edges. Four numbers per picture, fitted to all edge points with the
outliers left out, give where the edges are, the heading and the bend ahead without knowing the camera. A leak
through a faint line on one scan line does not move the fit the way it moved a per-line centre.

Where the car should be is defined against the right edge: the line it follows runs parallel to that edge, as
far from it as the car stood when the episode began (the simulator starts it in its lane). The simulator drops the
car onto the road, so the first half second is left out; the distance is the middle value of the dozen pictures
after that, each read on its own, because a single picture can be a third off. The left side is too changeable to
share that role — a worn line the camera barely shows, a shoulder, a bridge twice as wide as the lane — but its
points still help to find the heading and the bend. No training data and no map are involved, which is what lets
a collector start from nothing.

``RoadFinder`` returns the centre of that line near the car and further ahead, as fractions of half the picture
width, and the car's position across it; ``VisionExpert`` turns the two centres into a steering command. The
position is a picture measurement, not a ground truth: evaluation reports it as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class RoadView:
    found: bool
    near: float | None = None  # line to follow, close to the car: -1 left picture edge .. +1 right picture edge
    far: float | None = None  # the same line at the look-ahead row
    lateral: float | None = None  # car position across its lane: 0 on the line, +1 on the right edge, -1 as far left
    heading: float | None = None  # vanishing point of the road, same unit as near/far (positive: road leads right)
    curvature: float | None = None  # bend of the road ahead (positive: to the right), pixels at the look-ahead row
    rows: int = 0  # edge points the fit kept
    width: float | None = None  # lane width implied by the reference distance, pixels per row below the horizon
    runs: list = field(default_factory=list)  # (row, left, right, left seen, right seen) of every scan line
    model: tuple | None = None  # (c0, c2, c1 left, c1 right)


@dataclass
class RoadFinderSettings:
    # Row where the two edges of a straight, level road meet (the vanishing point). Measured on the simulator's
    # camera; it sits below the skyline because the camera looks slightly down.
    horizon: int = 73
    hood_rows: int = 12  # rows at the bottom that show the car itself
    top_margin: int = 5  # the rows right below the vanishing row are too compressed to measure
    row_step: int = 2  # distance between scan lines
    blur: int = 5  # Gaussian kernel that takes the speckle out of the asphalt before colours are compared
    # Colour distance (Lab a/b) from the surface in front of the car that ends the road. The camera's JPEG leaves
    # blocks of colour noise below this; a worn lane line can be fainter than that and is then simply not seen.
    chroma_tolerance: float = 8.
    texture_factor: float = 2.5  # ... or this many times the colour variation of the reference patch, if that is more
    bright_tolerance: float = 75.  # so does a surface this much lighter (a white kerb) ...
    dark_tolerance: float = 80.  # ... or this much darker (a wall in shade)
    edge_pixels: int = 2  # consecutive off-colour pixels that make an edge; a single one is noise
    patch_rows: int = 8  # size of the reference patch above the hood
    patch_cols: int = 40
    seed_search: int = 24  # columns searched either side when the expected centre is off-colour
    min_width: float = .05  # a scan line narrower than this fraction of the picture ends the road
    min_points: int = 3  # edge points a side needs to count as seen
    side_points: int = 6  # edge points each side needs before the distance between the edges is remembered
    trim: float = 3.  # an edge point this many robust standard deviations from its edge has no say in the fit
    prior_depth: int = 30  # row depth at which the priors below are expressed, in pixels
    curvature_prior: float = 1.  # weight of 'no bend' (one edge point's worth at prior_depth)
    width_prior: float = 100.  # weight of the remembered edge distance when one edge is out of view
    temporal_prior: float = .3  # weight of the previous picture's heading and bend (in edge points' worth)
    forget_after: int = 10  # pictures without a road after which the previous one no longer guides the search
    max_residual: float = 4.  # a fit that leaves a larger weighted RMS error (pixels) is not a road
    gate: float = 14.  # an edge point further than this (pixels) from the previous picture's edge is ignored
    look_ahead: float = .3  # the far centre is read this fraction of the way from the scan's top to the hood
    smoothing: float = .6  # weight of the newest picture in the reported centres
    surface_tolerance: float = 10.  # colour change under the car that needs a road with two clear edges to be believed
    support: float = .5  # share of the right-edge points of a picture that an accepted road has to explain
    jump: float = .6  # change of the right edge's slope between pictures that no sideways motion of the car explains
    clearance: float = .8  # smallest slope distance kept between the line to follow and the right edge
    snap: float = .5  # a reference this close to the episode's first one is taken to be that one
    settle: float = .02  # share of the remaining difference to the first reference removed per picture
    start_skip: int = 10  # pictures at the start of an episode left out of that distance: the car is still settling
    start_pictures: int = 12  # pictures after those whose right edge gives the distance to keep
    steady_pictures: int = 40  # accepted pictures in a row, no jump and at the episode's own distance: a steady view


class RoadFinder:
    def __init__(self, settings: RoadFinderSettings | None = None):
        self.settings = settings or RoadFinderSettings()
        self.reset()

    def reset(self) -> None:
        self.reference = None  # slope of the right edge when the episode began: the distance the car keeps from it
        self.edge_distance = None  # slope difference between the edges, from recent pictures that showed both
        self.smooth_near = self.smooth_far = None
        self.model = None  # the previous picture's (c0, c2, c1 left, c1 right)
        self.misses = 0
        self.road_colour = None  # Lab a/b of the surface the accepted roads were on
        self.patch_colour = None
        self.line = None  # slope of the line the car follows, as of the last accepted picture
        self.initial = None  # the reference the episode began with
        self.starts = []  # right-edge slopes of the episode's first pictures
        self.pending = None  # right-edge slope of a picture that suggested another edge and awaits a second one
        self.jumps = 0  # times the right edge was taken to be a different one
        self.pictures = 0  # pictures of this episode so far
        self.steady_run = 0  # accepted pictures in a row without a miss or a jump, at the episode's own distance

    @property
    def ready(self) -> bool:
        """The distance to keep has been taken: positions are measured from here on."""
        return len(self.starts) >= self.settings.start_pictures

    @property
    def steady(self) -> bool:
        """The road has been followed for a while without a doubt: no unreadable picture, no change of edge, the
        distance the episode began with. A collector disturbs the car only then."""
        return self.ready and self.steady_run >= self.settings.steady_pictures

    @property
    def width_per_row(self):
        """Lane width implied by the reference distance (the car in the middle), pixels per row."""
        return None if self.reference is None else 2 * self.reference

    # ---- picture to edge points -------------------------------------------------------------------
    def barrier(self, rgb: np.ndarray):
        """Pixels of the ground region that are not the surface in front of the car; ``None`` when that patch is
        not one surface (then there is no reference to trust)."""
        import cv2

        s = self.settings
        height, width = rgb.shape[:2]
        bottom = height - s.hood_rows
        top = s.horizon + s.top_margin
        ground = np.ascontiguousarray(rgb[top:bottom])
        if s.blur > 1:
            ground = cv2.GaussianBlur(ground, (s.blur, s.blur), 0)
        lab = cv2.cvtColor(ground, cv2.COLOR_RGB2LAB).astype(np.float32)
        patch = lab[-s.patch_rows:, width // 2 - s.patch_cols // 2: width // 2 + s.patch_cols // 2].reshape(-1, 3)
        reference = np.median(patch, axis=0)
        self.patch_colour = reference[1:3].astype(np.float64)
        spread = np.hypot(patch[:, 1] - reference[1], patch[:, 2] - reference[2])
        if float(spread.mean()) > s.chroma_tolerance:
            return None, top, bottom
        # A surface with a pattern of its own (the stones of the bridge) varies more than asphalt does: what
        # counts as another colour is measured against the variation of the patch itself.
        tolerance = max(s.chroma_tolerance, s.texture_factor * float(np.percentile(spread, 90)))
        difference = lab - reference
        off = ((np.hypot(difference[..., 1], difference[..., 2]) > tolerance)
               | (difference[..., 0] > s.bright_tolerance) | (difference[..., 0] < -s.dark_tolerance))
        # A 3x3 median drops single off-colour specks and keeps a line two pixels wide.
        off = cv2.medianBlur(off.astype(np.uint8), 3).astype(bool)
        return off, top, bottom

    def _predict(self, depth: float):
        """Road edges of the previous picture's model at ``depth`` rows below the vanishing row."""
        c0, c2, left, right = self.model
        return c0 + left * depth + c2 / depth, c0 + right * depth + c2 / depth

    def trace(self, off: np.ndarray, top: int, bottom: int):
        """One (row, left, right, left seen, right seen) per scan line, from the car upwards."""
        s = self.settings
        width = off.shape[1]
        need = max(1, s.edge_pixels)
        seed = width // 2
        runs = []
        for row in range(bottom - 1, top - 1, -s.row_step):
            line = off[row - top]
            if self.model is not None:
                left, right = self._predict(row - s.horizon)
                if right - left > 4:
                    seed = int(min(width - 1, max(0, round((left + right) / 2))))
            if line[seed]:
                low, high = max(0, seed - s.seed_search), min(width, seed + s.seed_search + 1)
                candidates = np.flatnonzero(~line[low:high])
                if not candidates.size:
                    break
                seed = int(low + candidates[np.argmin(np.abs(candidates + low - seed))])
            # An edge is ``need`` off-colour pixels in a row; ``solid[i]`` says such a stretch starts at column i.
            solid = line[:width - need + 1].copy()
            for shift in range(1, need):
                solid &= line[shift:width - need + 1 + shift]
            before = np.flatnonzero(solid[:max(0, seed - need + 1)])
            after = np.flatnonzero(solid[seed:])
            left_seen, right_seen = bool(before.size), bool(after.size)
            left = int(before[-1]) + need if left_seen else 0
            right = seed + int(after[0]) - 1 if right_seen else width - 1
            if right - left + 1 < s.min_width * width:
                break
            runs.append((int(row), left, right, left_seen, right_seen))
            seed = (left + right) // 2
        return runs

    # ---- edge points to a road ----------------------------------------------------------------------
    def fit(self, runs, *, gated: bool = True):
        """``(c0, c2, c1 left, c1 right), (left points kept, right points kept), rms`` or None.

        Least squares on the edge points, reweighted so that points far from their edge lose their say (a
        speckle of off-colour asphalt, a line too faint to stop the scan). The two edges have slopes of their
        own. Two soft priors keep the four numbers determined when the points alone do not: no bend unless the
        points show one, and, for an edge that is out of view, the distance between the edges remembered from
        the last pictures that showed both.
        """
        s = self.settings
        points = []  # (side, depth, column)
        previous = self.model if gated else None
        for row, left, right, left_seen, right_seen in runs:
            depth = float(row - s.horizon)
            expected = self._predict(depth) if previous is not None else None
            for side, seen, column in ((0, left_seen, left - .5), (1, right_seen, right + .5)):
                if not seen:
                    continue
                if expected is not None and abs(column - expected[side]) > s.gate + .15 * depth:
                    continue
                points.append((side, depth, column))
        if len(points) < s.min_points + 1:
            return None
        data = np.array(points, dtype=np.float64)
        sides, depths, columns = data[:, 0], data[:, 1], data[:, 2]
        counts = (int((sides == 0).sum()), int((sides == 1).sum()))
        design = np.column_stack([np.ones_like(depths), 1. / depths, np.where(sides == 0, depths, 0.),
                                  np.where(sides == 1, depths, 0.)])
        reference = float(s.prior_depth)
        rows, targets, strengths = [[0., 1. / reference, 0., 0.]], [0.], [s.curvature_prior]
        if previous is not None and s.temporal_prior > 0:
            # Heading and bend change little from one picture to the next. With one edge in view they are hard to
            # tell from that edge's slope, and this is what tells them apart.
            rows += [[1., 0., 0., 0.], [0., 1. / reference, 0., 0.]]
            targets += [previous[0], previous[1] / reference]
            strengths += [s.temporal_prior, s.temporal_prior]
        if min(counts) < s.min_points:
            if self.edge_distance is not None:
                rows.append([0., 0., -reference, reference])
                targets.append(reference * self.edge_distance)
            elif counts[0] < s.min_points:
                rows.append([0., 0., reference, reference])  # nothing known about the left edge: mirror the right
                targets.append(0.)
            else:
                return None  # the right edge is out of view and no distance to place it by
            strengths.append(s.width_prior)
        prior, prior_target = np.array(rows), np.array(targets)
        prior_weight = np.sqrt(np.array(strengths))
        weight = np.ones(len(data))
        solution = None
        for _ in range(8):
            root = np.sqrt(weight)
            stacked = np.vstack([design * root[:, None], prior * prior_weight[:, None]])
            solution = np.linalg.lstsq(stacked, np.concatenate([columns * root, prior_target * prior_weight]), rcond=None)[0]
            residual = columns - design @ solution
            scale = max(1.5, 1.4826 * float(np.median(np.abs(residual))))
            reach = np.abs(residual) / (s.trim * scale)
            weight = np.where(reach < 1., (1. - reach ** 2) ** 2, 0.)
        inside = weight > 0
        kept = (int((inside & (sides == 0)).sum()), int((inside & (sides == 1)).sum()))
        if sum(kept) < max(s.min_points + 1, len(data) // 2):
            return None
        rms = float(np.sqrt(np.sum(weight * residual ** 2) / np.sum(weight)))
        return tuple(float(v) for v in solution), kept, rms

    def _accept(self, fitted, runs, *, fresh: bool = False):
        """The fit if it is a road the car can follow, else None."""
        if fitted is None:
            return None
        s = self.settings
        (_c0, _c2, left, right), kept, rms = fitted
        seen_right = sum(1 for run in runs if run[4])
        if seen_right >= s.side_points and kept[1] < s.support * seen_right:
            # Most of what the scan found on the right is not where this road has its edge: the previous
            # picture's road is being stretched over a few stray points while the real edge is elsewhere.
            return None
        right_known = kept[1] >= s.min_points or (self.edge_distance is not None and kept[0] >= s.min_points)
        if rms > s.max_residual or not right_known or right <= left:
            return None
        if len(self.starts) < s.start_pictures and (kept[1] < s.min_points + 1 or right <= 0):
            return None  # the reference distance has to come from a right edge that is plainly there
        if fresh and kept[1] < s.side_points:
            return None  # a road found without guidance needs a right edge with plenty of points
        if self.road_colour is not None and min(kept) < s.min_points + 2 and \
                float(np.hypot(*(self.patch_colour - self.road_colour))) > s.surface_tolerance:
            # The car stands on something else than the road it was following, and what it sees is not a
            # road with two clear edges either: the verge, not a change of road surface.
            return None
        return fitted

    def __call__(self, rgb: np.ndarray) -> RoadView:
        s = self.settings
        width = rgb.shape[1]
        off, top, bottom = self.barrier(rgb)
        runs = self.trace(off, top, bottom) if off is not None else []
        self.pictures += 1
        starting = not self.ready
        # The first pictures are read one by one, without the previous picture's road as a guide: they are to be
        # independent measurements of where the car stands.
        fitted = self._accept(self.fit(runs, gated=not starting), runs) if runs else None
        if fitted is None and runs and self.model is not None and not starting:
            # The edges are not where the last picture's road says. Where a lane line ends and a wall begins the
            # edge really does jump: look again without the guidance of the previous road.
            fitted = self._accept(self.fit(runs, gated=False), runs, fresh=True)
        if fitted is None:
            self.misses += 1
            self.steady_run = 0
            if self.misses >= s.forget_after:
                self.model = None  # stop steering the search by a road that is no longer confirmed
            return RoadView(found=False, runs=runs)
        (c0, c2, left, right), kept, _rms = fitted
        jumped = False
        if starting:
            # The car stands where the simulator put it, and that is where it is to drive. The simulator drops it
            # onto the road: while it settles the camera is higher than it will be and every slope comes out up to
            # a third smaller, so those pictures are left out. One picture can still misjudge the distance to the
            # edge (a few points on the far kerb are enough); the middle value of a dozen does not.
            if self.pictures > s.start_skip:
                self.starts.append(right)
            self.reference = self.initial = float(np.median(self.starts)) if self.starts else right
        elif self.line is not None and abs(right - self.reference - self.line) > s.jump:
            # The car cannot move sideways this fast: the right edge is a different one now (the lane line gave way
            # to the kerb or the bridge wall), or this picture is a fluke. A second picture has to agree.
            if self.pending is None or abs(right - self.pending) > .5 * s.jump:
                self.pending = right
                self.model = (c0, c2, left, right)  # look for the edges where this picture has them
                self.misses += 1
                self.steady_run = 0
                return RoadView(found=False, runs=runs)
            # The line to follow goes on where it was, measured from the new edge. If that puts the car where
            # it stood against the first edge of the episode, it is that edge again.
            self.reference = right - self.line
            if abs(self.reference - self.initial) < s.snap:
                self.reference = self.initial
            self.jumps += 1
            jumped = True
        self.pending = None
        self.misses = 0
        self.model = (c0, c2, left, right)
        self.road_colour = self.patch_colour if self.road_colour is None else .7 * self.road_colour + .3 * self.patch_colour
        if min(kept) >= s.side_points:
            distance = right - left
            self.edge_distance = distance if self.edge_distance is None else .7 * self.edge_distance + .3 * distance
        if self.reference < self.initial or abs(self.reference - self.initial) < s.snap:
            # After a jump to a nearer edge (the bridge wall gives way to the lane line again) the car is closer
            # to its edge than it began: it moves back out, gently. Near its first value the reference settles
            # onto it, so the small errors made at the jumps do not add up over the laps. A reference larger than
            # the first one is left alone: beside a wall the car keeps its line instead of closing in.
            self.reference += s.settle * (self.initial - self.reference)
        self.reference = max(self.reference, s.clearance)
        at_home = abs(self.reference - self.initial) < s.snap
        self.steady_run = self.steady_run + 1 if not (starting or jumped) and at_home else 0
        half = width / 2
        # Slope of the line to follow, parallel to the right edge. While the reference is being taken the car is
        # on that line by definition.
        line = self.line = 0. if starting else right - self.reference

        def centre(depth: float) -> float:
            return (c0 + line * depth + c2 / depth) / half - 1

        near_depth = float(bottom - 1 - s.horizon)
        far_depth = float(top - s.horizon) + s.look_ahead * (bottom - 1 - top)
        near, far = centre(near_depth), centre(far_depth)
        if self.smooth_near is None:
            self.smooth_near, self.smooth_far = near, far
        else:
            self.smooth_near += s.smoothing * (near - self.smooth_near)
            self.smooth_far += s.smoothing * (far - self.smooth_far)
        return RoadView(found=True, near=float(self.smooth_near), far=float(self.smooth_far),
                        lateral=float(np.clip(-line / self.reference, -2., 2.)), heading=float(c0 / half - 1),
                        curvature=float(c2 / far_depth), rows=sum(kept), width=float(2 * self.reference), runs=runs,
                        model=(c0, c2, left, right))


class VisionExpert:
    """Steer towards the line ahead, with a share for the offset right in front of the car."""

    def __init__(self, far_gain: float = .8, near_gain: float = .3):
        if not all(math.isfinite(float(v)) for v in (far_gain, near_gain)) \
                or far_gain < 0 or near_gain < 0 or far_gain + near_gain <= 0:
            raise ValueError("gains must be non-negative and not both zero")
        self.far_gain, self.near_gain = float(far_gain), float(near_gain)

    def __call__(self, view: RoadView) -> float:
        return float(np.clip(self.far_gain * view.far + self.near_gain * view.near, -1., 1.))
