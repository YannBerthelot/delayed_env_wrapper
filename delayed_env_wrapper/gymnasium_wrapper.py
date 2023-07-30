from collections import deque
from typing import Any, SupportsFloat, TypeVar

import gymnasium as gym

from delayed_env_wrapper.errors import DelayError

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class ConstantDelayedWrapper(gym.Wrapper):
    def __init__(self, base_env: gym.Env, delay: int, reward: float = 0):
        if delay <= 0:
            raise DelayError(delay)
        gym.Wrapper.__init__(self, base_env)
        self._delay = delay
        self._reward = reward
        self._action_buffer = None

    @property
    def action_to_exec(self):
        if len(self._action_buffer) > 0:
            return self._action_buffer[0]
        raise ValueError("Buffer is not filled")

    @property
    def last_action_inserted(self):
        if len(self._action_buffer) > 0:
            return self._action_buffer[-1]
        raise ValueError("Buffer is not filled")

    @property
    def action_buffer(self):
        return self._action_buffer

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if len(self._action_buffer) == self._delay:
            actual_action = self._action_buffer.popleft()
            self._action_buffer.append(action)
            return self.env.step(actual_action)
        self._action_buffer.append(action)
        obs, info = self.reset(reset_buffer=False)
        return obs, self._reward, False, False, info

    def reset(self, reset_buffer=True, *args, **kwargs):
        if reset_buffer:
            self._action_buffer = deque([], maxlen=self._delay)
        return self.env.reset(*args, **kwargs)
