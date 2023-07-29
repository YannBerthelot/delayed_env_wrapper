from typing import Any, SupportsFloat, TypeVar

import gymnasium as gym
from collections import deque

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class ConstantDelayedWrapper(gym.Wrapper):
    def __init__(self, base_env, delay):
        gym.Wrapper.__init__(self, base_env)
        self._delay = delay
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

    def reset(self, *args, **kwargs):
        self._action_buffer = deque([], maxlen=self._delay)
        return self.env.reset(*args, **kwargs)
