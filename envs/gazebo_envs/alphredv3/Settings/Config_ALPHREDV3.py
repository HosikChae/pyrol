#!usr/bin/env python
from __future__ import absolute_import
__author__ = "Min Sung Ahn"
__email__ = "aminsung@gmail.com"
__copyright__ = "Copyright 2016 RoMeLa"
__date__ = "January 1, 1999"

__version__ = "0.0.1"
__status__ = "Prototype"

import sys

import numpy as np

from . import RobotData as RDS
from Settings.Macros_ALPHREDV3 import *

ROBOT = RDS.ALPHREDV3()
ROBOT.name = "ALPHRED"
ROBOT.gender = "Male"

# ===== Joints
j_id = [
		0,  # Body
		1,  2,  3,  4,  # Limb 1 (+X)
		5,  6,  7,  8,  # Limb 2 (+Y)
		9,  10, 11, 12, # Limb 3 (-X)
		13, 14, 15, 16, # Limb 4 (-Y)
		]

j_name = [
		'BODY',
		'LIMB1_HIP_YAW', 'LIMB1_HIP_PITCH', 'LIMB1_KNEE_PITCH', 'LIMB1_TOE',
		'LIMB2_HIP_YAW', 'LIMB2_HIP_PITCH', 'LIMB2_KNEE_PITCH', 'LIMB2_TOE',
		'LIMB3_HIP_YAW', 'LIMB3_HIP_PITCH', 'LIMB3_KNEE_PITCH', 'LIMB3_TOE',
		'LIMB4_HIP_YAW', 'LIMB4_HIP_PITCH', 'LIMB4_KNEE_PITCH', 'LIMB4_TOE',
		]

ROBOT.parse_joints(j_id, j_name)

# ===== Timing
ROBOT.dt = 0.001
ROBOT.freq = 1.0/ROBOT.dt

# ===== Kinematics and Joints
link_0 	= np.array([[0.0], [0.0], [0.0]])

# Limb 1 Dimensions
link_1 	= np.array([[BODY_OFFSET],  [0.0], [0.0]])
link_2 	= np.array([[ROTOR_OFFSET], [0.0], [0.0]])
link_3 	= np.array([[LENGTH_FEMUR], [0.0], [0.0]])
link_4 	= np.array([[LENGTH_TIBIA], [0.0], [0.0]])

# Limb 2 Dimensions
link_5 	= np.array([[0.0], [BODY_OFFSET],  [0.0]])
link_6 	= np.array([[0.0], [ROTOR_OFFSET], [0.0]])
link_7	= np.array([[0.0], [LENGTH_FEMUR], [0.0]])
link_8	= np.array([[0.0], [LENGTH_TIBIA], [0.0]])

# Limb 3 Dimensions
link_9	= np.array([[-BODY_OFFSET],  [0.0], [0.0]])
link_10	= np.array([[-ROTOR_OFFSET], [0.0], [0.0]])
link_11	= np.array([[-LENGTH_FEMUR], [0.0], [0.0]])
link_12	= np.array([[-LENGTH_TIBIA], [0.0], [0.0]])

# Limb 4 Dimensions
link_13	= np.array([[0.0], [-BODY_OFFSET],  [0.0]])
link_14	= np.array([[0.0], [-ROTOR_OFFSET], [0.0]])
link_15	= np.array([[0.0], [-LENGTH_FEMUR], [0.0]])
link_16	= np.array([[0.0], [-LENGTH_TIBIA], [0.0]])

# Body
ROBOT.add_joint(0, 'BODY', None, 1, link_0, 'Z')

# Limb 1
ROBOT.add_joint(1, 'LIMB1_HIP_YAW', 5, 2, link_1, 'Z', HIP_YAW_LIMIT_PLUS, HIP_YAW_LIMIT_MINUS)
ROBOT.add_joint(2, 'LIMB1_HIP_PITCH', None, 3, link_2, 'Y', HIP_PITCH_LIMIT_PLUS, HIP_PITCH_LIMIT_MINUS)
ROBOT.add_joint(3, 'LIMB1_KNEE_PITCH', None, 4, link_3, 'Y', KNEE_PITCH_LIMIT_PLUS, KNEE_PITCH_LIMIT_MINUS)
ROBOT.add_joint(4, 'LIMB1_TOE', None, None, link_4, 'Y')

# Limb 2
ROBOT.add_joint(5, 'LIMB2_HIP_YAW', 9, 6, link_5, 'Z', HIP_YAW_LIMIT_PLUS, HIP_YAW_LIMIT_MINUS)
ROBOT.add_joint(6, 'LIMB2_HIP_PITCH', None, 7, link_6, '-X', HIP_PITCH_LIMIT_PLUS, HIP_PITCH_LIMIT_MINUS)
ROBOT.add_joint(7, 'LIMB2_KNEE_PITCH', None, 8, link_7, '-X', KNEE_PITCH_LIMIT_PLUS, KNEE_PITCH_LIMIT_MINUS)
ROBOT.add_joint(8, 'LIMB2_TOE', None, None, link_8, '-X')

# Limb 3
ROBOT.add_joint(9, 'LIMB3_HIP_YAW', 13, 10, link_9, 'Z', HIP_YAW_LIMIT_PLUS, HIP_YAW_LIMIT_MINUS)
ROBOT.add_joint(10, 'LIMB3_HIP_PITCH', None, 11, link_10, '-Y', HIP_PITCH_LIMIT_PLUS, HIP_PITCH_LIMIT_MINUS)
ROBOT.add_joint(11, 'LIMB3_KNEE_PITCH', None, 12, link_11, '-Y', KNEE_PITCH_LIMIT_PLUS, KNEE_PITCH_LIMIT_MINUS)
ROBOT.add_joint(12, 'LIMB3_TOE', None, None, link_12, '-Y')

# Limb 4
ROBOT.add_joint(13, 'LIMB4_HIP_YAW', None, 14, link_13, 'Z', HIP_YAW_LIMIT_PLUS, HIP_YAW_LIMIT_MINUS)
ROBOT.add_joint(14, 'LIMB4_HIP_PITCH', None, 15, link_14, 'X', HIP_PITCH_LIMIT_PLUS, HIP_PITCH_LIMIT_MINUS)
ROBOT.add_joint(15, 'LIMB4_KNEE_PITCH', None, 16, link_15, 'X', KNEE_PITCH_LIMIT_PLUS, KNEE_PITCH_LIMIT_MINUS)
ROBOT.add_joint(16, 'LIMB4_TOE', None, None, link_16, 'X')

# Update all the kinematics parameters
ROBOT.init_setup(0)
ROBOT.IK()

# ===== Walking Parameters
