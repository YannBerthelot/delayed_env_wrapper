import gymnasium as gym
import numpy as np
import pytest

from delayed_env_wrapper.gymnasium_wrapper import ConstantDelayedWrapper


@pytest.fixture()
def setup_and_fill_buffer():
    base_env = gym.make("CartPole-v1")
    delay = 5
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay)
    delayed_env.reset()
    initial_actions = [0, 1, 0, 1, 0]
    for (
        action
    ) in (
        initial_actions
    ):  # action buffer for actions before the first agent selected action
        delayed_env.step(action)
    return delayed_env, initial_actions


def test_env_is_wrapper_of_base_env():
    base_env = gym.make("CartPole-v1")
    delay = 5
    delayed_env = ConstantDelayedWrapper(base_env, delay=delay)
    assert base_env.unwrapped == delayed_env.unwrapped


def test_fill_buffer_before_start(setup_and_fill_buffer):
    delayed_env, initial_actions = setup_and_fill_buffer
    assert np.array_equal(delayed_env.action_buffer, initial_actions)


def test_env_is_delayed(setup_and_fill_buffer):
    delayed_env, initial_actions = setup_and_fill_buffer
    actions = [1, 1, 0, 0, 1]
    for i, action in enumerate(actions):
        assert delayed_env.action_to_exec == initial_actions[i]
        delayed_env.step(action)
    assert np.array_equal(delayed_env.action_buffer, actions)


def test_reset_clears_buffer(setup_and_fill_buffer):
    delayed_env, initial_actions = setup_and_fill_buffer
    delayed_env.reset()
    assert len(delayed_env.action_buffer) == 0
