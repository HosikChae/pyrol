#!usr/bin/env python
__author__ = "Min Sung Ahn"
__email__ = "aminsung@gmail.com"
__copyright__ = "Copyright 2017 RoMeLa"
__date__ = "December 25, 2017"

__version__ = "0.0.1"
__status__ = "Prototype"

'''
MemoryManager is a macro file to your favorite memory segments.

Pre-generate your shared memory segments before proceeding with using them in the rest of your scripts. 
'''

import numpy as np
import time

import pyshmxtreme.SHMSegment as shmx
import pdb

import ctypes

class GPDData(ctypes.Structure):
    _fields_ = [
        ('left_horizontal',  ctypes.c_int), 
        ('left_vertical',    ctypes.c_int), 
        ('right_horizontal', ctypes.c_int), 
        ('right_vertical',   ctypes.c_int), 
        ('a_button',         ctypes.c_bool),
        ('b_button',         ctypes.c_bool),
        ('x_button',         ctypes.c_bool),
        ('y_button',         ctypes.c_bool),
        ('l1_button',        ctypes.c_bool),
        ('l2_button',        ctypes.c_bool),
        ('r1_button',        ctypes.c_bool),
        ('r2_button',        ctypes.c_bool),
        ('dpad',             ctypes.c_int)]

# ===== Create shared memory segments
# Time Data
TIME_STATE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='TIME_STATE', init=False)
TIME_STATE.add_blocks(name='time', data=np.zeros((1,1)))

FSM_STATE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='FSM_STATE', init=False)
FSM_STATE.add_blocks(name='state', data=np.zeros((1,1)))

# Joint State
JOINT_STATE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='JOINT_STATE', init=False)
JOINT_STATE.add_blocks(name='time_stamp', data=np.zeros((1, 1)))
JOINT_STATE.add_blocks(name='joint_positions', data=np.zeros((12, 1)))
JOINT_STATE.add_blocks(name='joint_velocities', data=np.zeros((12, 1)))
JOINT_STATE.add_blocks(name='foot_contacts', data=np.zeros((4, 1)))
JOINT_STATE.add_blocks(name='est_restart', data=np.zeros((1, 1)))
JOINT_STATE.add_blocks(name='error_state', data=np.zeros((12,1)))
JOINT_STATE.add_blocks(name='temperature', data=np.zeros((12,1)))
JOINT_STATE.add_blocks(name='current', data=np.zeros((12,1)))

# Simulation State
SIMULATION_STATE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='SIMULATION_STATE', init=False)
SIMULATION_STATE.add_blocks(name='sim_ready', data=np.zeros((1, 1)))

# Limb State
LIMB_STATE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='LIMB_STATE', init=False)
LIMB_STATE.add_blocks(name='time_stamp', data=np.zeros((1, 1)))
LIMB_STATE.add_blocks(name='mpc', data=np.zeros((1, 1)))
LIMB_STATE.add_blocks(name='limb1_pos', data=np.zeros((3, 1)))
LIMB_STATE.add_blocks(name='limb2_pos', data=np.zeros((3, 1)))
LIMB_STATE.add_blocks(name='limb3_pos', data=np.zeros((3, 1)))
LIMB_STATE.add_blocks(name='limb4_pos', data=np.zeros((3, 1)))

# MPC Communication used for simulation
MPC_COMMUNICATION = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='MPC_COMMUNICATION', init=False)
MPC_COMMUNICATION.add_blocks(name='mpc_wait', data=np.zeros((1, 1)))
MPC_COMMUNICATION.add_blocks(name='T1', data=np.zeros((2, 1)))
MPC_COMMUNICATION.add_blocks(name='T2', data=np.zeros((2, 1)))
MPC_COMMUNICATION.add_blocks(name='T3', data=np.zeros((2, 1)))
MPC_COMMUNICATION.add_blocks(name='T4', data=np.zeros((2, 1)))
MPC_COMMUNICATION.add_blocks(name='foot_contacts', data=np.zeros((4, 1)))


# MPC output
MPC_OUTPUT = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='MPC_OUTPUT', init=False)
MPC_OUTPUT.add_blocks(name='time_stamp', data=np.zeros((1,1)))
MPC_OUTPUT.add_blocks(name='solve_time', data=np.zeros((1,1)))
MPC_OUTPUT.add_blocks(name='swing_time', data=np.zeros((1,1)))
MPC_OUTPUT.add_blocks(name='stance_time', data=np.zeros((1,1)))
MPC_OUTPUT.add_blocks(name='h', data=np.zeros((1,1)))
MPC_OUTPUT.add_blocks(name='T1', data=np.zeros((2,1)))
MPC_OUTPUT.add_blocks(name='T2', data=np.zeros((2,1)))
MPC_OUTPUT.add_blocks(name='T3', data=np.zeros((2,1)))
MPC_OUTPUT.add_blocks(name='T4', data=np.zeros((2,1)))
MPC_OUTPUT.add_blocks(name='p1', data=np.zeros((4,1)))
MPC_OUTPUT.add_blocks(name='p2', data=np.zeros((4,1)))
MPC_OUTPUT.add_blocks(name='p3', data=np.zeros((4,1)))
MPC_OUTPUT.add_blocks(name='p4', data=np.zeros((4,1)))
MPC_OUTPUT.add_blocks(name='ax', data=np.zeros((5,1)))
MPC_OUTPUT.add_blocks(name='ay', data=np.zeros((5,1)))
MPC_OUTPUT.add_blocks(name='C1', data=np.zeros((1,1)))
MPC_OUTPUT.add_blocks(name='C2', data=np.zeros((1,1)))
MPC_OUTPUT.add_blocks(name='C3', data=np.zeros((1,1)))
MPC_OUTPUT.add_blocks(name='C4', data=np.zeros((1,1)))

# MPC Force
MPC_FORCE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='MPC_FORCE', init=False)
MPC_FORCE.add_blocks(name='time_stamp', data=np.zeros((1,1)))
MPC_FORCE.add_blocks(name='F1', data=np.zeros((3,1)))
MPC_FORCE.add_blocks(name='F2', data=np.zeros((3,1)))
MPC_FORCE.add_blocks(name='F3', data=np.zeros((3,1)))
MPC_FORCE.add_blocks(name='F4', data=np.zeros((3,1)))
MPC_FORCE.add_blocks(name='p1', data=np.zeros((3,1)))
MPC_FORCE.add_blocks(name='p2', data=np.zeros((3,1)))
MPC_FORCE.add_blocks(name='p3', data=np.zeros((3,1)))
MPC_FORCE.add_blocks(name='p4', data=np.zeros((3,1)))

# Joint Commands
JOINT_COMMANDS = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='JOINT_COMMANDS', init=False)
JOINT_COMMANDS.add_blocks(name='time_stamp', data=np.zeros((1, 1)))
JOINT_COMMANDS.add_blocks(name='coordinate_system', data=np.zeros((4,1)))  # used when when doing end effector force control
JOINT_COMMANDS.add_blocks(name='commands', data=np.zeros((12, 1)))
JOINT_COMMANDS.add_blocks(name='mode', data=-1*np.ones((1, 1)))
JOINT_COMMANDS.add_blocks(name='damping', data=np.zeros((1, 1)))
JOINT_COMMANDS.add_blocks(name='pause', data=np.zeros((1, 1)))
JOINT_COMMANDS.add_blocks(name='stop', data=np.zeros((1, 1)))
JOINT_COMMANDS.add_blocks(name='foot_contacts', data=np.zeros((4, 1)))
JOINT_COMMANDS.add_blocks(name='manipulation_mode', data=np.zeros((1, 1)))

# Joint Commands
WING_COMMANDS = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='WING_COMMANDS', init=False)
WING_COMMANDS.add_blocks(name='time_stamp', data=np.zeros((1, 1)))
WING_COMMANDS.add_blocks(name='commands', data=np.zeros((4, 1)))


# Estimator State
ESTIMATOR_STATE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='ESTIMATOR_STATE', init=False)
ESTIMATOR_STATE.add_blocks(name='time_stamp', data=np.zeros((1,1)))
ESTIMATOR_STATE.add_blocks(name='position', data=np.zeros((3, 1)))
ESTIMATOR_STATE.add_blocks(name='velocity', data=np.zeros((3, 1)))
ESTIMATOR_STATE.add_blocks(name='euler_ang', data=np.zeros((3, 1)))
ESTIMATOR_STATE.add_blocks(name='rot_matrix', data=np.zeros((9, 1)))
ESTIMATOR_STATE.add_blocks(name='ang_rate', data=np.zeros((3, 1)))
ESTIMATOR_STATE.add_blocks(name='est_return_status', data=np.zeros((1, 1)))

# Estimator Commands
ESTIMATOR_COMMANDS = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='ESTIMATOR_COMMANDS', init=False)
ESTIMATOR_COMMANDS.add_blocks(name='restart', data=np.zeros((1, 1)))

# Foot Contacts
FOOT_CONTACTS = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='FOOT_CONTACTS', init=False)
FOOT_CONTACTS.add_blocks(name='time_stamp', data=np.zeros((1, 1)))
FOOT_CONTACTS.add_blocks(name='contacts', data=np.zeros((4, 1)))

# Ipopt Thread status
TRAJECTORY_STATE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='TRAJECTORY_STATE', init=False)
TRAJECTORY_STATE.add_blocks(name='T_H', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='T_F1', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='T_F2', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='T_F3', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='T_F4', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='C_F1', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='C_F2', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='C_F3', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='C_F4', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='sh_F1', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='sh_F2', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='sh_F3', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='sh_F4', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='H', data=np.zeros((15,1)))
TRAJECTORY_STATE.add_blocks(name='F1', data=np.zeros((15,3)))
TRAJECTORY_STATE.add_blocks(name='F2', data=np.zeros((15,3)))
TRAJECTORY_STATE.add_blocks(name='F3', data=np.zeros((15,3)))
TRAJECTORY_STATE.add_blocks(name='F4', data=np.zeros((15,3)))
TRAJECTORY_STATE.add_blocks(name='wu_index', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='wf1_index', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='wf2_index', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='wf3_index', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='wf4_index', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='x', data=np.zeros((5000,1)))
TRAJECTORY_STATE.add_blocks(name='transition_time', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='num_phases', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='nvar', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='tp', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='T', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='new', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='heading', data=np.zeros((1,1)))
TRAJECTORY_STATE.add_blocks(name='step_height', data=np.zeros((1,1)))

# Main thread status
MAIN_THREAD = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='MAIN_THREAD', init=False)
MAIN_THREAD.add_blocks(name='traj_num', data=np.zeros((1,1)))
MAIN_THREAD.add_blocks(name='teleop_mode', data=np.zeros((1,1)))

# Nominal footsteps
NOMINAL_FOOTSTEPS = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='NOMINAL_FOOTSTEPS', init=False)
NOMINAL_FOOTSTEPS.add_blocks(name='F1', data=np.zeros((2,3)))
NOMINAL_FOOTSTEPS.add_blocks(name='F2', data=np.zeros((2,3)))
NOMINAL_FOOTSTEPS.add_blocks(name='F3', data=np.zeros((2,3)))
NOMINAL_FOOTSTEPS.add_blocks(name='F4', data=np.zeros((2,3)))
NOMINAL_FOOTSTEPS.add_blocks(name='ready', data=np.zeros((1,1)))

# Modified footsteps
MODIFIED_FOOTSTEPS = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='MODIFIED_FOOTSTEPS', init=False)
NOMINAL_FOOTSTEPS.add_blocks(name='F1', data=np.zeros((2,4)))
NOMINAL_FOOTSTEPS.add_blocks(name='F2', data=np.zeros((2,4)))
NOMINAL_FOOTSTEPS.add_blocks(name='F3', data=np.zeros((2,4)))
NOMINAL_FOOTSTEPS.add_blocks(name='F4', data=np.zeros((2,4)))
MODIFIED_FOOTSTEPS.add_blocks(name='ready', data=np.zeros((1,1)))

JOYSTICK_STATE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='JOYSTICK_STATE', init=False)
JOYSTICK_STATE.add_blocks(name='joystick_data', data=np.array(GPDData()))

DESIRED_STATE = shmx.SHMSegment(robot_name='ALPHREDV3', seg_name='DESIRED_STATE', init=False)
DESIRED_STATE.add_blocks(name='desired_velocity', data=np.zeros((3,1)))
DESIRED_STATE.add_blocks(name='desired_ang_rate', data=np.zeros((3,1)))

def init():
    '''Init if main'''
    TIME_STATE.initialize = True
    SIMULATION_STATE.initialize = True
    MPC_COMMUNICATION.initialize = True
    MPC_FORCE.initialize = True
    JOINT_STATE.initialize = True
    JOINT_COMMANDS.initialize = True
    WING_COMMANDS.initialize = True
    LIMB_STATE.initialize = True
    MPC_OUTPUT.initialize = True
    ESTIMATOR_STATE.initialize = True
    ESTIMATOR_COMMANDS.initialize = True
    TRAJECTORY_STATE.initialize = True
    MAIN_THREAD.initialize = True
    NOMINAL_FOOTSTEPS.initialize = True
    MODIFIED_FOOTSTEPS.initialize = True
    JOYSTICK_STATE.initialize = True
    DESIRED_STATE.initialize = True
    FSM_STATE.initialize = True
    FOOT_CONTACTS.initialize = True


def connect():
    '''Connect and create segment'''
    TIME_STATE.connect_segment()
    SIMULATION_STATE.connect_segment()
    MPC_COMMUNICATION.connect_segment()
    MPC_FORCE.connect_segment()
    JOINT_STATE.connect_segment()
    JOINT_COMMANDS.connect_segment()
    WING_COMMANDS.connect_segment()
    LIMB_STATE.connect_segment()
    MPC_OUTPUT.connect_segment()
    ESTIMATOR_STATE.connect_segment()
    ESTIMATOR_COMMANDS.connect_segment()
    TRAJECTORY_STATE.connect_segment()
    MAIN_THREAD.connect_segment()
    NOMINAL_FOOTSTEPS.connect_segment()
    MODIFIED_FOOTSTEPS.connect_segment()
    JOYSTICK_STATE.connect_segment()
    DESIRED_STATE.connect_segment()
    FSM_STATE.connect_segment()
    FOOT_CONTACTS.connect_segment()

if __name__ == '__main__':
    init()
    connect()
else:
    pass
