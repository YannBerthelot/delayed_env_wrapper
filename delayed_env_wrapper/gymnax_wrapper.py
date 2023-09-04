from collections import deque
from functools import partial
from typing import Optional, TypeVar, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from delayed_env_wrapper.errors import DelayError, FrameStackingError

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


@struct.dataclass
class EnvStateWithBuffer:
    state: environment.EnvState
    action_buffer: jnp.ndarray
    delay: int


@struct.dataclass
class EnvStateWithObservations:
    state: environment.EnvState
    observations: jnp.ndarray
    initial_observation: jnp.ndarray


def check_invalid_delays(delay, max_delay):
    if delay <= 0 or max_delay <= 0:
        raise DelayError(delay)


class ConstantDelayedWrapper(GymnaxWrapper):
    def __init__(
        self,
        base_env: environment.Environment,
        delay: int,
        max_delay: Optional[int] = None,
    ):
        GymnaxWrapper.__init__(self, base_env)
        self._delay = delay
        if max_delay is None:
            self._max_delay = delay
        else:
            self._max_delay = max_delay
        jax.debug.callback(check_invalid_delays, self._delay, self._max_delay)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state_with_buffer: EnvStateWithBuffer,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ):
        action_buffer = state_with_buffer.action_buffer
        state = state_with_buffer.state
        delay = state_with_buffer.delay
        buffer_size = jnp.sum(~jnp.isnan(action_buffer))

        def past_init(state, action_buffer, action, delay):
            """Process the action when the buffer is full"""
            action_buffer, actual_action = popleft_and_append_deque(
                action_buffer, action, idx=self._max_delay - delay + 1
            )
            n_obs, n_state, reward, done, info = self._env.step(
                key, state, actual_action, params
            )
            new_state_with_buffer = EnvStateWithBuffer(
                action_buffer=action_buffer, state=n_state, delay=delay
            )
            return n_obs, new_state_with_buffer, reward, done, info

        def pre_init(state, action_buffer, action, delay):
            """Process the action when the buffer is not yet full"""
            action_buffer, _ = popleft_and_append_deque(
                action_buffer, action, idx=self._max_delay - delay + 1
            )

            n_obs, _, reward, done, info = self._env.step(key, state, action, params)

            new_state_with_buffer = EnvStateWithBuffer(
                action_buffer=action_buffer, state=state, delay=delay
            )
            return (
                jnp.ones_like(n_obs),
                new_state_with_buffer,
                reward,
                done,
                info,
            )  # TODO return something else than ones? initial state?

        # jax.debug.breakpoint()
        return jax.lax.cond(
            (jnp.equal(buffer_size, delay)),
            past_init,
            pre_init,
            state,
            action_buffer,
            action,
            delay,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        action_buffer = jnp.ones(self._max_delay) * jnp.nan
        obs, state = self._env.reset(key, params)
        state_with_buffer = EnvStateWithBuffer(
            action_buffer=action_buffer, state=state, delay=self._delay
        )
        return obs, state_with_buffer


@jax.jit
def popleft_and_append_deque(jnp_deque: jnp.ndarray, new_value: float, idx: int = 1):
    """ "Sort of deque for jnp.arrays"""
    left_val = jnp_deque[0]
    jnp_deque = jnp_deque.at[:-1].set(jnp_deque[1:])
    jnp_deque = jnp_deque.at[-idx].set(new_value)
    return jnp_deque, left_val


def stack_observations(observations: jax.Array):
    """Stack the observations into a single observation"""
    return observations.flatten()


def get_repeated_obs(obs, n_times):
    return jnp.concatenate([obs for _ in range(n_times)])


class FrameStackingWrapper(GymnaxWrapper):
    def __init__(self, base_env: environment.Environment, num_of_frames: int):
        if num_of_frames <= 0:
            raise FrameStackingError(num_of_frames)
        GymnaxWrapper.__init__(self, base_env)
        self._num_of_frames = num_of_frames

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state_with_buffer: EnvStateWithObservations,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ):
        obs_buffer = state_with_buffer.observations
        state = state_with_buffer.state
        initial_obs = state_with_buffer.initial_observation
        buffer_size = jnp.floor_divide(
            jnp.sum(~jnp.isnan(obs_buffer)),
            self._env.observation_space(params).shape[0],
        )  # 4 is obs size

        def past_init(obs_buffer, action):
            """Process the action when the buffer is full"""
            n_obs, n_state, reward, done, info = self._env.step(
                key, state, action, params
            )
            obs_buffer, _ = popleft_and_append_deque(obs_buffer, n_obs)
            stacked_obs = stack_observations(obs_buffer)
            new_state_with_buffer = EnvStateWithObservations(
                observations=obs_buffer, state=n_state, initial_observation=initial_obs
            )
            return stacked_obs, new_state_with_buffer, reward, done, info

        def pre_init(obs_buffer, action):
            """Process the action when the buffer is not yet full"""
            n_obs, n_state, reward, done, info = self._env.step(
                key, state, action, params
            )
            obs_buffer = obs_buffer.at[buffer_size].set(n_obs)
            stacked_obs = stack_observations(obs_buffer)
            new_state_with_buffer = EnvStateWithObservations(
                observations=obs_buffer, state=n_state, initial_observation=initial_obs
            )
            n_obs = jnp.where(
                jnp.isnan(stacked_obs),
                get_repeated_obs(initial_obs, self._num_of_frames),
                stacked_obs,
            )
            return n_obs, new_state_with_buffer, reward, done, info

        return jax.lax.cond(
            (jnp.equal(buffer_size, self._num_of_frames)),
            past_init,
            pre_init,
            obs_buffer,
            action,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        obs_buffer = (
            jnp.ones(
                (self._num_of_frames, self._env.observation_space(params).shape[0])
            )
            * jnp.nan
        )
        obs, state = self._env.reset(key, params)
        state_with_buffer = EnvStateWithObservations(
            observations=obs_buffer, state=state, initial_observation=obs
        )
        return get_repeated_obs(obs, self._num_of_frames), state_with_buffer
