import jax
import gymnax
import numpy as np
import jax.numpy as jnp
import pytest
from delayed_env_wrapper.gymnax_wrapper import ConstantDelayedWrapper


@pytest.fixture()
def setup_and_fill_buffer():
    key = jax.random.PRNGKey(42)
    key_reset, key_step = jax.random.split(key)
    base_env, env_params = gymnax.make("CartPole-v1")
    delay = 5
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay)
    _, env_state, action_buffer = delayed_env.reset(key_reset, env_params)
    initial_actions = [0, 1, 0, 1, 0]
    for (
        action
    ) in (
        initial_actions
    ):  # action buffer for actions before the first agent selected action
        _, _, _, _, _, action_buffer = delayed_env.step(
            key_step, env_state, action, action_buffer, env_params
        )
    return delayed_env, initial_actions, env_params, env_state, key_step, action_buffer


def test_env_is_wrapper_of_base_env():
    base_env, env_params = gymnax.make("CartPole-v1")
    delay = 5
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay)
    assert isinstance(delayed_env._env, base_env.__class__)


def test_fill_buffer_before_start(setup_and_fill_buffer):
    delayed_env, initial_actions, _, _, _, action_buffer= setup_and_fill_buffer
    assert jnp.array_equal(
        action_buffer, jnp.array(initial_actions)
    )


def test_env_is_delayed(setup_and_fill_buffer):
    (
        delayed_env,
        initial_actions,
        env_params,
        env_state,
        key_step,
        action_buffer,
    ) = setup_and_fill_buffer
    actions = [1, 1, 0, 0, 1]
    for i, action in enumerate(actions):
        action_buffer[0] == initial_actions[i]
        action_buffer = delayed_env.step(
            key_step, env_state, action, action_buffer, env_params
        )[-1]
    assert jnp.array_equal(action_buffer, jnp.array(actions))
