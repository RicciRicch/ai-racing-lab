import numpy as np
import pygame
import pytest

from envs.track_geometry import (
    HEIGHT,
    TRACK_SETS,
    WIDTH,
    TrackGeometry,
    unseen_track,
)
from tracks import GRASS_COLOR, TRAINING_TRACKS


def pixel_is_grass(surface, x, y):
    return surface.get_at((x, y))[:3] == GRASS_COLOR


@pytest.mark.parametrize(
    "track",
    TRACK_SETS["all"],
    ids=lambda track: track.name
)
def test_mask_matches_rendered_pixels(track):
    surface = pygame.Surface((WIDTH, HEIGHT))
    track.draw(surface)

    geometry = TrackGeometry(track)

    rng = np.random.default_rng(0)

    for x, y in zip(
        rng.integers(0, WIDTH, 2000),
        rng.integers(0, HEIGHT, 2000)
    ):
        assert geometry.grass[x, y] == pixel_is_grass(surface, x, y)


@pytest.mark.parametrize(
    "track",
    TRACK_SETS["all"],
    ids=lambda track: track.name
)
def test_start_position_is_on_road(track):
    geometry = TrackGeometry(track)

    assert not geometry.is_blocked(track.start_x, track.start_y)


def test_out_of_bounds_is_blocked():
    geometry = TrackGeometry(TRAINING_TRACKS[0])

    assert geometry.is_blocked(-1, 10)
    assert geometry.is_blocked(10, HEIGHT)

    blocked = geometry.blocked_array(
        np.array([-5, WIDTH, 500]),
        np.array([10, 10, 590])
    )

    assert blocked.tolist() == [True, True, False]


def test_unseen_track_copy_matches_evaluation_script():
    # Importing the script opens a (dummy) window and defines draw_track().
    import evaluate_unseen_track as original

    original.draw_track()

    original_grass = np.all(
        pygame.surfarray.array3d(original.screen) == np.array(GRASS_COLOR),
        axis=2
    )

    geometry = TrackGeometry(unseen_track)

    assert np.array_equal(geometry.grass, original_grass)
    assert unseen_track.checkpoints == original.checkpoints
    assert (unseen_track.start_x, unseen_track.start_y) == (
        original.START_X,
        original.START_Y,
    )


@pytest.mark.parametrize(
    "track",
    TRACK_SETS["all"],
    ids=lambda track: track.name
)
def test_car_does_not_collide_at_spawn(track):
    from envs.env_car import EnvCar

    car = EnvCar(
        track.start_x,
        track.start_y,
        track.start_angle,
        TrackGeometry(track),
    )

    car.check_track_collision()

    assert car.alive


@pytest.mark.parametrize(
    "track",
    TRACK_SETS["all"],
    ids=lambda track: track.name
)
def test_every_checkpoint_is_reachable(track):
    # Count car positions that touch each checkpoint without colliding.
    # The training tracks allow 49-69 positions per checkpoint; the
    # original unseen track allowed only 4-5 for its side checkpoints,
    # which made a full lap practically impossible.
    from envs.env_car import EnvCar

    car = EnvCar(0, 0, 0, TrackGeometry(track))

    for index, checkpoint in enumerate(track.checkpoints):
        vertical = checkpoint.height > checkpoint.width
        reachable = set()

        for x in range(checkpoint.left - 40, checkpoint.right + 40):
            for y in range(checkpoint.top - 40, checkpoint.bottom + 40):
                for angle in (0, 180, -90, 90, 45, -45, 135, -135):
                    car.x, car.y, car.angle = x, y, angle
                    car.alive = True
                    car.check_track_collision()

                    if car.alive and car.get_rect().colliderect(checkpoint):
                        reachable.add(x if vertical else y)

        assert len(reachable) >= 40, f"checkpoint {index}"
