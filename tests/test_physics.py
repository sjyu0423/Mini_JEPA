import numpy as np

from mini_jepa.physics import SimulationConfig, simulate_trajectory


def test_simulation_is_deterministic() -> None:
    config = SimulationConfig(num_frames=20)
    first = simulate_trajectory(config, seed=123)
    second = simulate_trajectory(config, seed=123)
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.velocities, second.velocities)
    np.testing.assert_array_equal(first.collision_counts, second.collision_counts)


def test_elastic_reflection_preserves_speed_and_reverses_component() -> None:
    config = SimulationConfig(
        num_frames=2,
        timestep=0.5,
        ball_radius=0.1,
        min_speed=0.0,
        max_speed=0.0,
    )
    trajectory = simulate_trajectory(
        config,
        seed=0,
        initial_position=(0.7, 0.5),
        initial_velocity=(1.0, 0.25),
    )
    np.testing.assert_allclose(trajectory.positions[1], (0.6, 0.625))
    np.testing.assert_allclose(trajectory.velocities[1], (-1.0, 0.25))
    np.testing.assert_array_equal(trajectory.collision_counts[1], (1, 0))
    np.testing.assert_allclose(
        np.linalg.norm(trajectory.velocities[0]), np.linalg.norm(trajectory.velocities[1])
    )


def test_ball_always_remains_inside_arena() -> None:
    config = SimulationConfig(num_frames=500, timestep=0.15, ball_radius=0.07)
    trajectory = simulate_trajectory(config, seed=81)
    assert np.all(trajectory.positions >= config.ball_radius)
    upper = np.asarray(config.arena_size) - config.ball_radius
    assert np.all(trajectory.positions <= upper)


def test_high_speed_motion_resolves_multiple_collisions() -> None:
    config = SimulationConfig(
        num_frames=2,
        timestep=1.0,
        ball_radius=0.1,
        min_speed=0.0,
        max_speed=0.0,
    )
    trajectory = simulate_trajectory(
        config,
        seed=0,
        initial_position=(0.5, 0.5),
        initial_velocity=(20.0, -13.0),
    )
    assert np.all(trajectory.positions[1] >= 0.1)
    assert np.all(trajectory.positions[1] <= 0.9)
    assert np.all(trajectory.collision_counts[1] > 1)
    np.testing.assert_allclose(np.abs(trajectory.velocities[1]), np.abs(trajectory.velocities[0]))


def test_corner_collision_reflects_both_axes() -> None:
    config = SimulationConfig(
        num_frames=2,
        timestep=0.5,
        ball_radius=0.1,
        min_speed=0.0,
        max_speed=0.0,
    )
    trajectory = simulate_trajectory(
        config,
        seed=0,
        initial_position=(0.5, 0.5),
        initial_velocity=(1.0, 1.0),
    )
    np.testing.assert_allclose(trajectory.positions[1], (0.8, 0.8))
    np.testing.assert_allclose(trajectory.velocities[1], (-1.0, -1.0))
    np.testing.assert_array_equal(trajectory.collision_counts[1], (1, 1))
