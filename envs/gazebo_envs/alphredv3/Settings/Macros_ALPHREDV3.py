#!usr/bin/env python
from __future__ import division
from past.utils import old_div
__author__ 		= "Jeff Yu"
__email__ 		= "c.jeffyu@gmail.com"
__copyright__ 	= "Copyright 2017 RoMeLa"
__date__ 		= "June 9, 2017"

__version__ 	= "1.0.0"
__status__ 		= "Prototype"

"""
Script that holds useful macros that can be used to link a joint name to its joint id, for example.
"""

# --------------------------------
# ALPHRED JOINTS:
# Import these by doing 'from Macros_ALPHRED import *'
# Use these when accessing a joint in the robot
# --------------------------------
BODY 				= 0

LIMB1_HIP_YAW 		= 1
LIMB1_HIP_PITCH 	= 2
LIMB1_KNEE_PITCH 	= 3
LIMB1_TOE 			= 4

LIMB2_HIP_YAW 		= 5
LIMB2_HIP_PITCH 	= 6
LIMB2_KNEE_PITCH 	= 7
LIMB2_TOE 			= 8

LIMB3_HIP_YAW 		= 9
LIMB3_HIP_PITCH 	= 10
LIMB3_KNEE_PITCH 	= 11
LIMB3_TOE 			= 12

LIMB4_HIP_YAW 		= 13
LIMB4_HIP_PITCH 	= 14
LIMB4_KNEE_PITCH 	= 15
LIMB4_TOE 			= 16

"""
Constants
"""
GRAV = 9.81
PI = 3.1415926535897932
"""
Static Parameters
"""
# Link Lengths in (m)
LENGTH_FEMUR	= 0.350
LENGTH_TIBIA	= 0.34915
BODY_OFFSET  	= 0.1375
ROTOR_OFFSET	= 0.1025 	# From CAD: directly in the center with the robot sprawled out. No z offset between any joints

# Motor Constant (Nm/A)
MOTOR_CONST = 0.88 	# New from Tym
INV_MOTOR_CONST = old_div(1,MOTOR_CONST)

# Torque Limits
#TORQUE_MAX = 3.0
#TORQUE_MIN = -3.0
TORQUE_MAX = 52.8
TORQUE_MIN = -52.8
#TORQUE_MAX = 30.0
#TORQUE_MIN = -30.0
MAX_IQ = TORQUE_MAX*INV_MOTOR_CONST

# Encoder Offsets
VELRATE = 4000.0
ENCODER_COUNTS = 262144
ENC2RAD = old_div(2*PI,ENCODER_COUNTS)
RAD2ENC = 1.0/ENC2RAD

# Joint Limits
HIP_YAW_LIMIT_PLUS = 1.57075
HIP_YAW_LIMIT_MINUS = -1.57075

HIP_PITCH_LIMIT_PLUS = 1.70
HIP_PITCH_LIMIT_MINUS = -1.70

KNEE_PITCH_LIMIT_PLUS = 2.938
KNEE_PITCH_LIMIT_MINUS = -2.938

# Motor Gains
# Hip Yaw
HIP_YAW_POS_P = 0.005
HIP_YAW_POS_I = 0.00
HIP_YAW_POS_D = 1.0

HIP_YAW_VEL_P = 0.200
HIP_YAW_VEL_I = 0.01
HIP_YAW_VEL_D = 0.0

HIP_YAW_FOR_P = 0.01
HIP_YAW_FOR_I = 0.0
HIP_YAW_FOR_D = 0.6

# Hip Pitch
HIP_PITCH_POS_P = 0.005
HIP_PITCH_POS_I = 0.00
HIP_PITCH_POS_D = 1.0

HIP_PITCH_VEL_P = 0.200
HIP_PITCH_VEL_I = 0.01
HIP_PITCH_VEL_D = 0.00

HIP_PITCH_FOR_P = 0.01
HIP_PITCH_FOR_I = 0.00
HIP_PITCH_FOR_D = 0.6

# Knee Pitch
KNEE_PITCH_POS_P = 0.005
KNEE_PITCH_POS_I = 0.00
KNEE_PITCH_POS_D = 1.0

KNEE_PITCH_VEL_P = 0.20
KNEE_PITCH_VEL_I = 0.01
KNEE_PITCH_VEL_D = 0.00

KNEE_PITCH_FOR_P = 0.01
KNEE_PITCH_FOR_I = 0.00
KNEE_PITCH_FOR_D = 0.6

# IQ Gains
IQ_P = 0.001
IQ_I = 0.0001
IQ_D = 0.00

# ID Gains
ID_P = 0.001
ID_I = 0.0001
ID_D = 0.00

# Gazebo Simulation PID Gains
GAZEBO_HIP_YAW_P = 100.0
GAZEBO_HIP_YAW_I = 0.0
GAZEBO_HIP_YAW_D = 1.0

GAZEBO_HIP_PITCH_P = 100.0
GAZEBO_HIP_PITCH_I = 0.0
GAZEBO_HIP_PITCH_D = 1.0

GAZEBO_KNEE_PITCH_P = 100.0
GAZEBO_KNEE_PITCH_I = 0.0
GAZEBO_KNEE_PITCH_D = 1.0