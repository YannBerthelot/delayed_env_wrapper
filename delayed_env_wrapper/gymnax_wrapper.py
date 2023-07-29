from collections import deque
from functools import partial
from typing import Optional, TypeVar, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


@struct.dataclass
class EnvStateWithBuffer:
    state: environment.EnvState
    action_buffer: jnp.ndarray

class ConstantDelayedWrapper(GymnaxWrapper):
    def __init__(self, base_env, delay):
        GymnaxWrapper.__init__(self, base_env)
        self._delay = delay

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
        buffer_size = jnp.sum(~jnp.isnan(action_buffer))

        def past_init(action_buffer, action):
            action_buffer, actual_action = popleft_and_append_deque(
                action_buffer, action
            )
            n_obs, n_state, reward, done, info = self._env.step(
                key, state, actual_action, params
            )
            new_state_with_buffer = EnvStateWithBuffer(action_buffer=action_buffer, state=n_state)
            return n_obs, new_state_with_buffer, reward, done, info

        def pre_init(action_buffer, action):
            action_buffer = action_buffer.at[buffer_size].set(action)
            n_obs, _, reward, done, info = self._env.step(key, state, action, params)
            new_state_with_buffer = EnvStateWithBuffer(action_buffer=action_buffer, state=state)
            return jnp.ones_like(n_obs), new_state_with_buffer, reward, done, info

        return jax.lax.cond(
            (jnp.equal(buffer_size, self._delay)),
            past_init,
            pre_init,
            action_buffer,
            action,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        action_buffer = jnp.ones(self._delay) * jnp.nan
        obs, state = self._env.reset(key, params)
        state_with_buffer = EnvStateWithBuffer(action_buffer=action_buffer, state=state)
        return obs, state_with_buffer


@jax.jit
def popleft_and_append_deque(jnp_deque, new_value):
    """ "Sort of deque for jnp.arrays"""
    left_val = jnp_deque[0]
    jnp_deque = jnp_deque.at[:-1].set(jnp_deque[1:])
    jnp_deque = jnp_deque.at[-1].set(new_value)
    return jnp_deque, left_val
