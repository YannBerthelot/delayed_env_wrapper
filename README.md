# Delayed-Environment Wrapper

This package proposes a wrapper around RL environments, either in the gymnasium or gymnax format, that allows to handle delayed action-execution in environments.
This type of environments aims at modeling real life interactions between agent and environment in which the actions are not directly effective, or their action is not instantly measured (which covers almost every industrial application).

## Installation
Will be available on PyPi soon, in the meantime :
```bash
pip install git+https://github.com/YannBerthelot/delayed_env_wrapper.git
```

## Usage

### Gymnasium
You just have to specify the delay. The actions that are given to the environment at time t will automatically only be effective at time t+delay. 
A delay of 0 is equivalent to the base environment.
```python
import gymnasium as gym

from delayed_env_wrapper.gymnasium_wrapper import ConstantDelayedWrapper

base_env = gym.make("CartPole-v1")
delayed_env = ConstantDelayedWrapper(base_env, delay=5)

obs, info = delayed_env.reset()
obs, reward, terminated, truncated, info = delayed_env.step(action)
```
### Gymnax
You just have to specify the delay.
```python
import gymnax

from delayed_env_wrapper.gymnax_wrapper import ConstantDelayedWrapper

base_env, env_params = gymnax.make("CartPole-v1")
delayed_env = ConstantDelayedWrapper(base_env, delay=5)

obs, env_state = delayed_env.reset(key_reset, env_params)
n_obs, n_state, reward, done, info = delayed_env.step(key_step, env_state, action, env_params)
```
### Initial actions
Given that you have delay between the first timestep (for which you select an action) and when it affects the environment, actions need to be decided for the first d-timesteps (with d being the delay). To do so the environment will return the initial observation and reward until the first action had effect. Another possibility (NOT IMPLEMENTED YET) is to provide a serie of actions at the environment setup that will be executed while waiting for the first actual action to affect the environment.


# To-do

-[ ] Allow to provide pre-decided actions
-[ ] Allow to decide what reward to return if not providing actions.