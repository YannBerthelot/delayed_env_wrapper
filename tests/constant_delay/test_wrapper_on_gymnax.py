import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from delayed_env_wrapper.gymnax_wrapper import ConstantDelayedWrapper


@pytest.fixture()
def setup_and_fill_buffer():
    key = jax.random.PRNGKey(42)
    key_reset, key_step = jax.random.split(key)
    base_env, env_params = gymnax.make("CartPole-v1")
    delay = 5
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay)
    _, env_state  = delayed_env.reset(key_reset, env_params)
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


def test_fill_buffer_before_start(setup_and_fill_buffer):
    delayed_env, initial_actions, _, env_state, _ = setup_and_fill_buffer
    action_buffer = env_state.action_buffer
    assert jnp.array_equal(action_buffer, jnp.array(initial_actions))


def test_env_is_delayed(setup_and_fill_buffer):
    (
        delayed_env,
        initial_actions,
        env_params,
        env_state,
        key_step,
    ) = setup_and_fill_buffer
    action_buffer = env_state.action_buffer
    actions = [1, 1, 0, 0, 1]
    for i, action in enumerate(actions):
        assert action_buffer[0] == initial_actions[i]
        _, env_state, _, _, _ = delayed_env.step(
            key_step, env_state, action, env_params
        )
        action_buffer = env_state.action_buffer
    assert jnp.array_equal(action_buffer, jnp.array(actions))
