import time

from pybear import Manager
import numpy as np

from pyrol.envs.real.bear_pendulum.bear_pendulum import BearPendulumEnv
from pyrol.envs.real.bear_pendulum import constants
from dev.gabe.test4_colin import TD3

env = BearPendulumEnv(port='/dev/ttyUSB0',max_torque=1.4, theta0_range=(-0.6, 0.6))
env.connect()

env.stop()
print(env.observe_states())
while True:
    print(env.observe_states())
    time.sleep(0.01)