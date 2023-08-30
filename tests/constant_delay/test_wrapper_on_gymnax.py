import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from delayed_env_wrapper.errors import DelayError
from delayed_env_wrapper.gymnax_wrapper import (
    ConstantDelayedWrapper,
    FrameStackingWrapper,
)


@pytest.fixture()
def setup_and_fill_delay_buffer():
    key = jax.random.PRNGKey(42)
    key_reset, key_step = jax.random.split(key)
    base_env, env_params = gymnax.make("CartPole-v1")
    delay = 5
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay)
    _, env_state = delayed_env.reset(key_reset, env_params)
    initial_actions = [0, 1, 0, 1, 0]
    for (
        action
    ) in (
        initial_actions
    ):  # action buffer for actions before the first agent selected action
        _, env_state, _, _, _ = delayed_env.step(
            key_step, env_state, action, env_params
        )
    return delayed_env, initial_actions, env_params, env_state, key_step


def test_env_is_wrapper_of_base_env():
    base_env, env_params = gymnax.make("CartPole-v1")
    delay = 5
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay)
    assert isinstance(delayed_env._env, base_env.__class__)


def test_fill_buffer_before_start(setup_and_fill_delay_buffer):
    delayed_env, initial_actions, _, env_state, _ = setup_and_fill_delay_buffer
    action_buffer = env_state.action_buffer
    assert jnp.array_equal(action_buffer, jnp.array(initial_actions))


def test_env_is_delayed(setup_and_fill_delay_buffer):
    (
        delayed_env,
        initial_actions,
        env_params,
        env_state,
        key_step,
    ) = setup_and_fill_delay_buffer
    action_buffer = env_state.action_buffer
    actions = [1, 1, 0, 0, 1]
    for i, action in enumerate(actions):
        assert action_buffer[0] == initial_actions[i]
        _, env_state, _, _, _ = delayed_env.step(
            key_step, env_state, action, env_params
        )
        action_buffer = env_state.action_buffer
    assert jnp.array_equal(action_buffer, jnp.array(actions))


def test_invalid_delays():
    base_env, env_params = gymnax.make("CartPole-v1")
    delay = 0
    with pytest.raises(DelayError):
        ConstantDelayedWrapper(base_env, delay=delay)
    delay = -1
    with pytest.raises(DelayError):
        ConstantDelayedWrapper(base_env, delay=delay)


def get_n_first_obs_with_0_policy(
    key_step, base_env, env_params, num_of_frames, key_reset
):
    """Get the first num_of_frames observations from the environment when always taking 0 action"""
    action = 0
    observations = []
    obs, env_state = base_env.reset(key_reset, env_params)
    observations.append(obs)
    for _ in range(num_of_frames - 1):
        obs, env_state, _, _, _ = base_env.step(key_step, env_state, action, env_params)
        observations.append(obs)
    return observations


@pytest.fixture()
def setup_stack_buffer():
    key = jax.random.PRNGKey(42)
    key_reset, key_step = jax.random.split(key)
    base_env, env_params = gymnax.make("CartPole-v1")
    num_of_frames = 5
    expected_observations = get_n_first_obs_with_0_policy(
        key_step, base_env, env_params, num_of_frames + 2, key_reset
    )
    stacked_env = FrameStackingWrapper(base_env, num_of_frames=num_of_frames)
    init_obs, env_state = stacked_env.reset(key_reset, env_params)

    return (
        stacked_env,
        env_params,
        env_state,
        key_step,
        expected_observations,
        init_obs,
        num_of_frames,
    )


def test_env_is_stacked(setup_stack_buffer):
    """Check that the env has the correct frames stacked"""
    (
        stacked_env,
        env_params,
        env_state,
        key_step,
        expected_observations,
        init_obs,
        num_of_frames,
    ) = setup_stack_buffer

    action = 0

    def get_repeated_obs(obs, n_times):
        return jnp.concatenate([obs for _ in range(n_times)])

    expected_intermediate_observations = {
        0: get_repeated_obs(expected_observations[0], num_of_frames),
        1: jnp.concatenate(
            [
                expected_observations[1],
                get_repeated_obs(expected_observations[0], num_of_frames - 1),
            ]
        ).flatten(),
        2: jnp.concatenate(
            [
                expected_observations[1],
                expected_observations[2],
                get_repeated_obs(expected_observations[0], num_of_frames - 2),
            ]
        ).flatten(),
        3: jnp.concatenate(
            [
                expected_observations[1],
                expected_observations[2],
                expected_observations[3],
                get_repeated_obs(expected_observations[0], num_of_frames - 3),
            ]
        ).flatten(),
        4: jnp.concatenate(
            [
                expected_observations[1],
                expected_observations[2],
                expected_observations[3],
                expected_observations[4],
                get_repeated_obs(expected_observations[0], num_of_frames - 4),
            ]
        ).flatten(),
        5: jnp.concatenate(
            [
                expected_observations[1],
                expected_observations[2],
                expected_observations[3],
                expected_observations[4],
                expected_observations[5],
            ]
        ).flatten(),
        6: jnp.concatenate(
            [
                expected_observations[2],
                expected_observations[3],
                expected_observations[4],
                expected_observations[5],
                expected_observations[6],
            ]
        ).flatten(),
    }

    assert jnp.array_equal(init_obs, expected_intermediate_observations[0])
    for i in range(1, num_of_frames + 2):
        obs, env_state, _, _, _ = stacked_env.step(
            key_step, env_state, action, env_params
        )
        assert jnp.array_equal(obs, expected_intermediate_observations[i])
