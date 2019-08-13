# Environments
*Descriptions of the available custom environments.*
## Environment List
### ALPHREDv3
Name: Alphred version 3 (ALPHRED/ ALPHREDv3)  
Description: 4-legged, radially symmetric robot with point feet  
Simulator: Gazebo  
Physics Engine: \[Need to find out and add here\]  
Other:  

### \[ADD ANY ADDITONAL ENVIRONMENTS HERE\]

## Instruction For New Customized Environments
* Folder Structure  
```bash
gym-foo/
       README.md  
       setup.py  
       gym_foo/
              __init__.py  
              envs/
                     __init__.py  
                     foo_env.py
```

* `gym-foo/setup.py` should have the following lines:
```python
from setuptools import setup

setup(name='gym_foo',
      version='0.0.1',
      install_requires=['gym']#And any other dependencies required
)
```
* `gym-foo/gym_foo/__init__.py` should have the following:
```python
from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_foo.envs:FooEnv',
)
```
The `id` variable is what gym.make() uses to call the environment
* `gym-foo/gym_foo/envs/__init__.py` inits the environment
```python
from gym_foo.gym_foo.envs.foo_env import FooEnv
```
* Actual environment details in `gym-foo/gym_foo/envs/foo_env.py`:
```python
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    ...
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human', close=False):
    ...
```
test
