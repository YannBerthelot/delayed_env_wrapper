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
        self._action_buffer = deque([None for _ in range(delay)], maxlen=delay)
        self._action_to_exec = None

    @property
    def action_to_exec(self):
        return self._action_buffer[0]

    @property
    def last_action_inserted(self):
        return self._action_buffer[-1]

    @property
    def action_buffer(self):
        return self._action_buffer

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        actual_action = self._action_buffer.popleft()
        self._action_buffer.append(action)
        if self.env._elapsed_steps >= self._delay:
            return self.env.step(actual_action)
