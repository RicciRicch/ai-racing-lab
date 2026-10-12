"""Headless track geometry for the reinforcement-learning environment.

The original simulation detects grass by reading pixels from the
rendered pygame screen (``screen.get_at``). That works for real-time
visualization but is slow for RL, which needs hundreds of thousands of
simulation steps without a window.

``TrackGeometry`` draws a track once onto an off-screen surface using
the same ``Track.draw`` function as the game, then stores the result as
a boolean NumPy mask. Sensor and collision queries become array lookups
while producing exactly the same answers as the pixel-based version.
"""

import numpy as np
import pygame

from tracks import GRASS_COLOR, ROAD_COLOR, TRAINING_TRACKS, Track


# Same window size as main.py and the evaluation scripts.
WIDTH = 1000
HEIGHT = 700


# ==================================================
# UNSEEN TRACK
# ==================================================
#
# Copy of the track defined in evaluate_unseen_track.py.
# That script opens a window as soon as it is imported, so the
# definition is repeated here. tests/test_track_geometry.py verifies
# that both versions produce an identical grass mask.

def draw_unseen_track(screen):
    # Outer road shape
    pygame.draw.ellipse(screen, ROAD_COLOR, (110, 90, 780, 520))

    # Inner grass creates a narrower road
    pygame.draw.ellipse(screen, GRASS_COLOR, (330, 215, 340, 270))

    # Side narrowing sections
    pygame.draw.rect(screen, GRASS_COLOR, (110, 250, 90, 200))
    pygame.draw.rect(screen, GRASS_COLOR, (800, 250, 90, 200))


unseen_track = Track(
    name="Unseen Narrow Oval",

    start_x=500,
    start_y=600,
    start_angle=-90,

    # Clockwise: bottom-right, right, top-right,
    # top-left, left, bottom-left
    checkpoints=[
        pygame.Rect(650, 500, 50, 20),
        pygame.Rect(800, 310, 20, 60),
        pygame.Rect(650, 150, 50, 20),
        pygame.Rect(300, 150, 50, 20),
        pygame.Rect(180, 310, 20, 60),
        pygame.Rect(300, 500, 50, 20),
    ],

    draw_function=draw_unseen_track
)


UNSEEN_TRACKS = [
    unseen_track
]

TRACK_SETS = {
    "train": TRAINING_TRACKS,
    "unseen": UNSEEN_TRACKS,
    "all": TRAINING_TRACKS + UNSEEN_TRACKS,
}


# ==================================================
# GEOMETRY
# ==================================================

class TrackGeometry:
    """A track rasterized into a grass mask indexed as ``mask[x, y]``."""

    def __init__(
        self,
        track,
        width=WIDTH,
        height=HEIGHT
    ):
        self.track = track
        self.width = width
        self.height = height

        surface = pygame.Surface(
            (width, height)
        )

        track.draw(surface)

        # surfarray uses (x, y) indexing, matching screen.get_at((x, y)).
        pixels = pygame.surfarray.array3d(
            surface
        )

        self.grass = np.all(
            pixels == np.array(GRASS_COLOR),
            axis=2
        )

    @property
    def checkpoints(self):
        return self.track.checkpoints

    def is_blocked(self, x, y):
        """Same rule as Car.check_collision: off-screen or grass."""
        if (
            x < 0
            or x >= self.width
            or y < 0
            or y >= self.height
        ):
            return True

        return bool(
            self.grass[x, y]
        )

    def blocked_array(self, xs, ys):
        """Vectorized ``is_blocked`` for integer coordinate arrays."""
        outside = (
            (xs < 0)
            | (xs >= self.width)
            | (ys < 0)
            | (ys >= self.height)
        )

        inside_x = np.clip(xs, 0, self.width - 1)
        inside_y = np.clip(ys, 0, self.height - 1)

        return outside | self.grass[inside_x, inside_y]


_geometry_cache = {}


def get_geometry(track):
    """Rasterize each track once per process."""
    key = id(track)

    if key not in _geometry_cache:
        _geometry_cache[key] = TrackGeometry(
            track
        )

    return _geometry_cache[key]
