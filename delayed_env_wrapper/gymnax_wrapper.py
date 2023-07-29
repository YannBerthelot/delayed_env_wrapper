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


class ConstantDelayedWrapper(GymnaxWrapper):
    def __init__(self, base_env, delay):
        GymnaxWrapper.__init__(self, base_env)
        self._delay = delay

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        action_buffer: jnp.ndarray,
        params: Optional[environment.EnvParams] = None,
    ):
        buffer_size = jnp.sum(~jnp.isnan(action_buffer))

        def past_init(action_buffer, action):
            action_buffer, actual_action = popleft_and_append_deque(
                action_buffer, action
            )
            n_obs, n_state, reward, done, info = self._env.step(
                key, state, actual_action, params
            )
            return n_obs, n_state, reward, done, info, action_buffer

        def pre_init(action_buffer, action):
            action_buffer = action_buffer.at[buffer_size].set(action)
            n_obs, _, reward, done, info = self._env.step(key, state, action, params)
            return jnp.ones_like(n_obs), state, reward, done, info, action_buffer

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
        return obs, state, action_buffer


@jax.jit
def popleft_and_append_deque(jnp_deque, new_value):
    """ "Sort of deque for jnp.arrays"""
    left_val = jnp_deque[0]
    jnp_deque = jnp_deque.at[:-1].set(jnp_deque[1:])
    jnp_deque = jnp_deque.at[-1].set(new_value)
    return jnp_deque, left_val
