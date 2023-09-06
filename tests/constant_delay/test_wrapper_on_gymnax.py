import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from delayed_env_wrapper.errors import DelayError
from delayed_env_wrapper.gymnax_wrapper import (
    ConstantDelayedWrapper,
    FrameStackingWrapper,
    AugmentedObservationWrapper,
)

from gymnax.environments.classic_control.cartpole import EnvState


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
    key = jax.random.PRNGKey(42)
    key_reset, key_step = jax.random.split(key)
    base_env, base_env_params = gymnax.make("CartPole-v1")
    _, base_env_state = base_env.reset(key_reset, base_env_params)

    for i, action in enumerate(actions):
        assert action_buffer[0] == initial_actions[i]
        _, env_state, _, _, _ = delayed_env.step(
            key_step, env_state, action, env_params
        )
        _, base_env_state, _, _, _ = base_env.step(
            key_step, base_env_state, initial_actions[i], base_env_params
        )
        assert base_env_state == env_state.state
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


def get_obs_from_actions(actions, key_step, env, env_params, key_reset):
    """Get the first num_of_frames observations from the environment when always taking 0 action"""
    observations = []
    obs, env_state = env.reset(key_reset, env_params)
    observations.append(obs)
    for action in actions:
        obs, env_state, _, _, _ = env.step(key_step, env_state, action, env_params)
        observations.append(obs)
    return observations


def test_action_stacking():
    key = jax.random.PRNGKey(42)
    key_reset, key_step = jax.random.split(key)
    base_env, env_params = gymnax.make("CartPole-v1")
    obs_dim = base_env.observation_space(env_params).shape[0]
    delay = 5
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay)
    action_stacked_env = AugmentedObservationWrapper(delayed_env, num_of_frames=delay)
    actions = [1, 0, 1, 0, 1]
    expected_base_observations = get_obs_from_actions(
        actions, key_step, delayed_env, env_params, key_reset
    )
    observations = get_obs_from_actions(
        actions, key_step, action_stacked_env, env_params, key_reset
    )
    expected_actions = {
        0: jnp.zeros(delay),
        1: jnp.concatenate([jnp.zeros(delay - 1), jnp.array(actions[:1])]),
        2: jnp.concatenate([jnp.zeros(delay - 2), jnp.array(actions[:2])]),
        3: jnp.concatenate([jnp.zeros(delay - 3), jnp.array(actions[:3])]),
        4: jnp.concatenate([jnp.zeros(delay - 4), jnp.array(actions[:4])]),
        5: jnp.array(actions),
    }
    for i, (obs, expected_base_obs) in enumerate(
        zip(observations, expected_base_observations)
    ):
        assert obs.shape[0] == obs_dim + delay
        assert jnp.array_equal(expected_base_obs, obs[:obs_dim])
        assert jnp.array_equal(expected_actions[i], obs[obs_dim:])


def test_action_stacking_has_correct_obs_space():
    base_env, env_params = gymnax.make("CartPole-v1")
    obs_dim = base_env.observation_space(env_params).shape[0]
    delay = 5
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay)
    action_stacked_env = AugmentedObservationWrapper(delayed_env, num_of_frames=delay)
    assert action_stacked_env.observation_space(env_params).shape == (obs_dim + delay,)


def test_frame_stacking_has_correct_obs_space():
    base_env, env_params = gymnax.make("CartPole-v1")
    obs_dim = base_env.observation_space(env_params).shape[0]
    delay = 5
    stacked_env = FrameStackingWrapper(base_env, num_of_frames=delay)
    assert stacked_env.observation_space(env_params).shape == (obs_dim + delay,)


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


def test_env_is_stacked_for_obs(setup_stack_buffer):
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


MAX_DELAY = 3


def reset_delayed_env_and_params(delay):
    base_env, env_params = gymnax.make("CartPole-v1")
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay, max_delay=MAX_DELAY)
    key_reset = jax.random.PRNGKey(42)
    _, env_state = delayed_env.reset(key_reset, env_params)


def get_expected_actions(t):
    expected_actions = [
        jnp.array(
            [
                jnp.array([1.0, jnp.nan, jnp.nan]),
                jnp.array([jnp.nan, 1.0, jnp.nan]),
                jnp.array([jnp.nan, jnp.nan, 1.0]),
            ]
        ),
        jnp.array(
            [
                jnp.array([0.0, jnp.nan, jnp.nan]),
                jnp.array([1.0, 0.0, jnp.nan]),
                jnp.array([jnp.nan, 1.0, 0.0]),
            ]
        ),
        jnp.array(
            [
                jnp.array([1.0, jnp.nan, jnp.nan]),
                jnp.array([0.0, 1.0, jnp.nan]),
                jnp.array([1.0, 0.0, 1.0]),
            ]
        ),
    ]
    return expected_actions[t]


def get_env_states(actions):
    key = jax.random.PRNGKey(42)
    key_reset, key_step = jax.random.split(key)
    base_env, base_env_params = gymnax.make("CartPole-v1")
    _, base_env_state = base_env.reset(key_reset, base_env_params)
    env_states = [base_env_state]
    for action in actions:
        _, base_env_state, _, _, _ = base_env.step(
            key_step, base_env_state, action, base_env_params
        )
        env_states.append(base_env_state)
    return env_states


from delayed_env_wrapper.gymnax_wrapper import EnvStateWithBuffer

import numpy as np


def merge_expected_states(expected_states):
    keys = [x for x in expected_states[0].__dataclass_fields__.keys()]
    state = {}
    for key in keys:
        state[key] = (
            np.concatenate(
                [ex_state.__dict__[key].reshape(-1, 1) for ex_state in expected_states]
            )
            .reshape(1, -1)
            .flatten()
        )
    return EnvState(**state)


def get_expected_states(t):
    actions = [1, 0, 1]
    env_states = get_env_states(actions)

    expected_states = [
        [env_states[0], env_states[0], env_states[0]],
        [env_states[1], env_states[0], env_states[0]],
        [env_states[2], env_states[1], env_states[0]],
        [env_states[3], env_states[2], env_states[1]],
    ]

    return expected_states[t]


def reset_and_step_delayed_env(delay):
    base_env, env_params = gymnax.make("CartPole-v1")
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay, max_delay=MAX_DELAY)
    key = jax.random.PRNGKey(42)
    key_reset, key_step = jax.random.split(key)
    _, env_state = delayed_env.reset(key_reset, env_params)
    # pre-init
    states = []
    buffers = []

    for action in [1, 0, 1]:
        _, env_state, _, _, _ = delayed_env.step(
            key_step, env_state, action, env_params
        )

        states.append(env_state.state)
        buffers.append(env_state.action_buffer)
    return states, buffers


def test_reset_vectorized_env_with_multiple_delays():
    delays = jnp.array([1, 2, 3])
    jax.vmap(reset_delayed_env_and_params, in_axes=(0))(delays)


def check_env_states_are_equal(env_state_1, env_state_2):
    for key in env_state_1.__dict__.keys():
        if not jnp.array_equal(env_state_1.__dict__[key], env_state_2.__dict__[key]):
            return False
    return True


def test_step_vectorized_env_with_multiple_delays():
    delays = jnp.array([1, 2, 3])
    states, buffers = jax.vmap(reset_and_step_delayed_env, in_axes=(0))(delays)
    for i, buffer in enumerate(buffers):
        assert jnp.array_equal(buffer, get_expected_actions(i), equal_nan=True)
    for i, state in enumerate(states):
        assert check_env_states_are_equal(
            state, merge_expected_states(get_expected_states(i))
        )
