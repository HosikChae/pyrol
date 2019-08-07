#!usr/bin/env python
from __future__ import absolute_import
__author__ = "Min Sung Ahn"
__email__ = "aminsung@gmail.com"
__copyright__ = "Copyright 2016 RoMeLa"
__date__ = "January 1, 1999"

__version__ = "0.0.1"
__status__ = "Prototype"

"""
General configuration settings that are independent of the robot model.
"""

import os

import time
import re

from . import Config_ALPHREDV3


# Simulation True / False ?
simulation = True
# Choose Simulator Vrep / Gazebo ?
simulator_name = 'Gazebo'

Body = Config_ALPHREDV3.ROBOT
ROBOT_NAME = Body.name
ROBOT_GENDER = Body.gender

# Control Parameters
DT = Body.dt
sim_dt = Body.dt

# pyBEAR settings
baudrate = 8000000
pdm_port1 = '/dev/AL3L1'
pdm_port2 = '/dev/AL3L2'
pdm_port3 = '/dev/AL3L3'
pdm_port4 = '/dev/AL3L4'

# Turn on safe mode to use joint limits
Body.safe_mode = False
# Turn on threshold mode for joint limits
Body.th_mode = True

# True if GDP is being used to control the robot.
Body.GDP = False

# Setup scripts
if simulation == True:
    Body.simulation = True
    Body.simulator_name = simulator_name
    Body.sim_dt = sim_dt
    Body.sim_freq = 1.0/Body.sim_dt
    Body.sim_synchronous = True
else:
    Body.simulation = False
    Body.simulator_name = simulator_name
