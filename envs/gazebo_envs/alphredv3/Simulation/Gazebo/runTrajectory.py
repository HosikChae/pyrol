from builtins import range
from gazebopy import RobotInterface as GazeboRobotInterface
import time
import numpy as np

NUM_LIMBS = 4
NUM_JOINTS_PER_LIMB = 3
NUM_JOINTS = NUM_LIMBS*NUM_JOINTS_PER_LIMB

HIP_YAW_P = 100.0
HIP_YAW_I = 0.0
HIP_YAW_D = 1.0

HIP_PITCH_P = 100.0
HIP_PITCH_I = 0.0
HIP_PITCH_D = 1.0

KNEE_PITCH_P = 100.0
KNEE_PITCH_I = 0.0
KNEE_PITCH_D = 1.0


simulator = GazeboRobotInterface('ALPHRED', NUM_JOINTS)
simulator.reset_simulation()
simulator.unpause_physics()
simulator.set_step_size(0.001)
simulator.set_operating_mode(GazeboRobotInterface.POSITION_PID_MODE)

p_gains = [HIP_YAW_P, HIP_PITCH_P, KNEE_PITCH_P]*NUM_LIMBS
i_gains = [HIP_YAW_I, HIP_PITCH_I, KNEE_PITCH_I]*NUM_LIMBS
d_gains = [HIP_YAW_D, HIP_PITCH_D, KNEE_PITCH_D]*NUM_LIMBS

simulator.set_all_position_pid_gains(p_gains, i_gains, d_gains)

trajectory = np.genfromtxt('data.csv', delimiter=',')
rows,cols = np.shape(trajectory)

state_data = np.zeros((1+3*NUM_JOINTS+3,cols))

for step in range(cols):
    position_commands = [trajectory[0,step], trajectory[1,step], trajectory[2,step]]*NUM_LIMBS
    simulator.set_command_positions(position_commands)
    state_data[0,step] = simulator.get_current_time()
    state_data[1:NUM_JOINTS+1,step] = simulator.get_current_position().transpose()
    state_data[NUM_JOINTS+1:2*NUM_JOINTS+1,step] = simulator.get_current_velocity().transpose()
    state_data[2*NUM_JOINTS+1:3*NUM_JOINTS+1,step] = simulator.get_current_force().transpose()
    state_data[3*NUM_JOINTS+1:3*NUM_JOINTS+3+1,step] = simulator.get_imu_acceleration().transpose()
    time.sleep(.01)

# np.savetxt("output_data.csv", state_data, delimiter=",")

time.sleep(4)
simulator.set_all_position_pid_gains([0.0]*NUM_JOINTS, [0.0]*NUM_JOINTS, [0.0]*NUM_JOINTS)
simulator.set_command_positions([0.0]*NUM_JOINTS)