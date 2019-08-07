 #!usr/bin/env python
from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from past.utils import old_div
__author__ = "Jeffrey Yu"
__email__ = "c.jeffyu@gmail.com"
__copyright__ = "Copyright 2017 RoMeLa"
__date__ = "October 30, 2017"

__version__ = "0.0.1"
__status__ = "Prototype"

import collections
import pdb

# import Library.imu.LMS_3DM_GX4_25 as imu
import numpy as np
from ..Util import MathFcn as mathfcn
# import Util.MathFcn as mathfcn
from ..Settings.Macros_ALPHREDV3 import *
import time
from ..Util import MemoryManager as MM
# import Util.MemoryManager as MM
# from pybear import Manager
from ..Util import MathFcn as MF
# import Util.MathFcn as MF
from numba import njit
from ..Util import Dynamics_and_Kinematics as DK
# import Util.Dynamics_and_Kinematics as DK

#from Simulation.Vrep.VrepInterface import RobotInterface as VrepInterface

try:
    from ..Simulation.Vrep.VrepInterface import RobotInterface as VrepInterface
except:
    print("Failed to import vrep interface.")
    pass

try:
    from ..Simulation.Gazebo.ALPHREDGazeboInterface import ALPHREDGazeboInterface
except:
    print("Failed to import gazebo interface.")
    pass


class RobotDataStruct(object):
    def __init__(self):
        # Personal info
        self.name = None
        self.gender = None

        self.joint = collections.defaultdict(lambda: collections.defaultdict(int))
        self.joint_no = None
        self.ft = collections.defaultdict(lambda: collections.defaultdict(int))

        # Rotational variables
        self.rx = 0.0  # Roll
        self.ry = 0.0  # Pitch
        self.rz = 0.0  # Yaw
        self.drx = 0.0  # angular rate about x
        self.dry = 0.0  # angular rate about y
        self.drz = 0.0  # angular rate about z
        self.R = np.zeros((3, 3))  # Rotation matrix

        # Timing
        self.dt = None
        self.freq = None
        self.sim_dt = None
        self.sim_freq = None
        self.motor_com_dt = None

        # Useful variables
        self._aX = np.array([[1], [0], [0]])
        self._aY = np.array([[0], [1], [0]])
        self._aZ = np.array([[0], [0], [1]])
        self.save_data = np.empty([10000, 25])
        self.data_index = 0

        # Other settings
        self.simulation = False  # Boolean flag controlled by Settings.Config. Could be '== None' too.
        self.simulator = None
        self.simulator_name = None
        self.safe_mode = False  # Boolean flag controlled by Robot.py
        self.th_mode = False
        self.saving_data = False
        self.GDP = False      # Boolean that signals if the GDP is being used to control the robot

        # IMU settings
        self.running_imu = False
        self.imu_port = None
        self.sim_imu = None

        # Foot contact
        self.foot1 = False
        self.foot2 = False
        self.foot3 = False
        self.foot4 = False

        # Flag for manipulation mode
        self.manipulation = False

        self.mpc = False  # signals to the MPC

        MM.connect()

    def add_joint(self, j_id, name, sister, child, np_b, let_a, max=np.pi, min=-np.pi, q=0, dq=0, u=0):
        self.joint_axis_vec = {'X': self._aX, 'Y': self._aY, 'Z': self._aZ, \
                               '-X': -self._aX, '-Y': -self._aY, '-Z': -self._aZ}
        try:
            if (let_a.upper() not in self.joint_axis_vec):
                raise JointAxisError
            if np_b.shape != (3, 1):
                raise JointRelativePositionError
            a = self.joint_axis_vec[let_a.upper()]
            b = np_b
        except JointAxisError:
            raise JointAxisError("Your joint axis vector must be either X, Y, or Z")
        except JointRelativePositionError:
            raise JointRelativePositionError("Your relative position must be a np.ndarray of size/shape (3,1)")

        joint_info = {'id': j_id, 'parent': None, 'name': name, 'sister': sister, 'child': child, 'b': b, 'a': a,
                      'max': max, 'min': min, 'q': q, 'dq': dq, 'u': u}
        self.joint[j_id].update(joint_info)
        # self.joint[name].update(joint_info)
        # self._add_by_id(j_id, name, sister, child, b, a, q)
        self.joints_no = old_div(len(self.joint), 2)

    def _add_by_id(self, j_id, name, sister, child, b, a, q, u):
        # self.joint[j_id] = {'id': j_id, 'parent': None, 'name': name, 'sister': sister, 'child': child, 'b': b, 'a': a, 'q': q}
        self.joint[j_id]['id'] = j_id
        self.joint[j_id]['name'] = name
        self.joint[j_id]['sister'] = sister
        self.joint[j_id]['child'] = child
        self.joint[j_id]['b'] = b
        self.joint[j_id]['a'] = a
        self.joint[j_id]['q'] = q
        self.joint[j_id]['u'] = u

    def parse_joints(self, j_id, name):
        for idx in range(len(j_id)):
            self.joint[j_id[idx]]['name'] = name[idx]
        # self.joint[name[idx]]['id'] = j_id[idx] # Maybe unnecessary

    def init_setup(self, j_start):
        self._hierarchy_init(j_start)

    def _hierarchy_init(self, j_start):
        self._hierarchy_update(j_start)
        for j_id in self.joint:
            self.joint[j_id]['p'] = np.array([[0.0], [0.0], [0.0]])  # Should be moved to robot specific
            self.joint[j_id]['R'] = np.eye(3)

    def _hierarchy_update(self, j_id):
        if j_id == 0:
            self.joint[j_id]['parent'] = None
        if self.joint[j_id]['child'] != None:
            self.joint[self.joint[j_id]['child']].update({'parent': j_id})
            self._hierarchy_update(self.joint[j_id]['child'])
        if self.joint[j_id]['sister'] != None:
            self.joint[self.joint[j_id]['sister']]['parent'] = self.joint[j_id]['parent']
            self._hierarchy_update(self.joint[j_id]['sister'])

    def FK(self):
        Q = np.zeros((12, 1))
        n = 0
        for j_id in self.joint:
            if not j_id % 4 == 0:
                Q[n][0] = self.joint[j_id]['q']
                n = n + 1

        limbs = DK.forward_kinematics(Q)

        self.joint[LIMB1_TOE]['p'] = limbs[:, 0].reshape(3, 1) + self.joint[0]['p']
        self.joint[LIMB2_TOE]['p'] = limbs[:, 1].reshape(3, 1) + self.joint[0]['p']
        self.joint[LIMB3_TOE]['p'] = limbs[:, 2].reshape(3, 1) + self.joint[0]['p']
        self.joint[LIMB4_TOE]['p'] = limbs[:, 3].reshape(3, 1) + self.joint[0]['p']


class ALPHREDV3(RobotDataStruct):

    def __init__(self):
        RobotDataStruct.__init__(self)

        self.modes = {"torque": 0, "velocity": 1, "position": 2, "direct_position": 3, "ee_force": 10}

    def IK(self):
        r = self.joint[LIMB1_TOE]['p'] - self.joint[BODY]['p']
        q1, q2, q3 = DK.inverse_kinematics(r,1)
        self.joint[LIMB1_HIP_YAW]['q'] = q1
        self.joint[LIMB1_HIP_PITCH]['q'] = q2
        self.joint[LIMB1_KNEE_PITCH]['q'] = q3

        r = self.joint[LIMB2_TOE]['p'] - self.joint[BODY]['p']
        q4, q5, q6 = DK.inverse_kinematics(r,2)
        self.joint[LIMB2_HIP_YAW]['q'] = q4
        self.joint[LIMB2_HIP_PITCH]['q'] = q5
        self.joint[LIMB2_KNEE_PITCH]['q'] = q6

        r = self.joint[LIMB3_TOE]['p'] - self.joint[BODY]['p']
        q7, q8, q9 = DK.inverse_kinematics(r,3)
        self.joint[LIMB3_HIP_YAW]['q'] = q7
        self.joint[LIMB3_HIP_PITCH]['q'] = q8
        self.joint[LIMB3_KNEE_PITCH]['q'] = q9

        r = self.joint[LIMB4_TOE]['p'] - self.joint[BODY]['p']
        q10, q11, q12 = DK.inverse_kinematics(r,4)
        self.joint[LIMB4_HIP_YAW]['q'] = q10
        self.joint[LIMB4_HIP_PITCH]['q'] = q11
        self.joint[LIMB4_KNEE_PITCH]['q'] = q12

    def set_command_position(self):
        commands = {}
        commands['time_stamp'] = np.array([[time.time()]])
        commands['coordinate_system'] = np.zeros((4,1))
        commands['damping'] = np.array([[0.0]])
        commands['mode'] = np.array([[self.modes['direct_position']]])
        commands['pause'] = np.array([[0.0]])
        commands['stop'] = np.array([[0.0]])
        commands['foot_contacts'] = np.array([[self.foot1], [self.foot2], [self.foot3], [self.foot4]])
        commands['manipulation_mode'] = np.array([[self.manipulation]])
        commands['commands'] = np.array([[self.joint[LIMB1_HIP_YAW]['q']],
                                         [self.joint[LIMB1_HIP_PITCH]['q']],
                                         [self.joint[LIMB1_KNEE_PITCH]['q']],
                                         [self.joint[LIMB2_HIP_YAW]['q']],
                                         [self.joint[LIMB2_HIP_PITCH]['q']],
                                         [self.joint[LIMB2_KNEE_PITCH]['q']],
                                         [self.joint[LIMB3_HIP_YAW]['q']],
                                         [self.joint[LIMB3_HIP_PITCH]['q']],
                                         [self.joint[LIMB3_KNEE_PITCH]['q']],
                                         [self.joint[LIMB4_HIP_YAW]['q']],
                                         [self.joint[LIMB4_HIP_PITCH]['q']],
                                         [self.joint[LIMB4_KNEE_PITCH]['q']]])
        if self.simulation:
            while True:
                sim = MM.SIMULATION_STATE.get()
                if sim['sim_ready'][0,0] == 1.0:
                    break
        MM.JOINT_COMMANDS.set(commands)

    def set_command_torque(self):
        commands = {}
        commands['time_stamp'] = np.array([[time.time()]])
        commands['coordinate_system'] = np.zeros((4,1))
        commands['damping'] = np.array([[0.0]])
        commands['mode'] = np.array([[self.modes['torque']]])
        commands['pause'] = np.array([[0.0]])
        commands['stop'] = np.array([[0.0]])
        commands['foot_contacts'] = np.array([[self.foot1], [self.foot2], [self.foot3], [self.foot4]])
        commands['manipulation_mode'] = np.array([[self.manipulation]])
        commands['commands'] = np.array([[self.joint[LIMB1_HIP_YAW]['u']],
                                         [self.joint[LIMB1_HIP_PITCH]['u']],
                                         [self.joint[LIMB1_KNEE_PITCH]['u']],
                                         [self.joint[LIMB2_HIP_YAW]['u']],
                                         [self.joint[LIMB2_HIP_PITCH]['u']],
                                         [self.joint[LIMB2_KNEE_PITCH]['u']],
                                         [self.joint[LIMB3_HIP_YAW]['u']],
                                         [self.joint[LIMB3_HIP_PITCH]['u']],
                                         [self.joint[LIMB3_KNEE_PITCH]['u']],
                                         [self.joint[LIMB4_HIP_YAW]['u']],
                                         [self.joint[LIMB4_HIP_PITCH]['u']],
                                         [self.joint[LIMB4_KNEE_PITCH]['u']]])
        if self.simulation:
            while True:
                sim = MM.SIMULATION_STATE.get()
                if sim['sim_ready'][0,0] == 1.0:
                    break
        MM.JOINT_COMMANDS.set(commands)


    def set_command_eeforces(self, coordinate_system, forces):
        # Tells the motor controller the desired end effector forces.
        # coordinate_system: (4x1) vector signalling if in the robot frame (0) or in the inertial frame (1)
        # forces: (12x1) vector with the desired end effector forces limb 1 are the first 3 entries and so on and so forth
        commands = {}
        commands['time_stamp'] = np.array([[time.time()]])
        commands['coordinate_system'] = coordinate_system
        commands['damping'] = np.array([[0.0]])
        commands['mode'] = np.array([[self.modes['ee_force']]])
        commands['pause'] = np.array([[0.0]])
        commands['stop'] = np.array([[0.0]])
        commands['foot_contacts'] = np.array([[self.foot1], [self.foot2], [self.foot3], [self.foot4]])
        commands['manipulation_mode'] = np.array([[self.manipulation]])
        commands['commands'] = forces

        if self.simulation:
            while True:
                sim = MM.SIMULATION_STATE.get()
                if sim['sim_ready'][0,0] == 1.0:
                    break
        MM.JOINT_COMMANDS.set(commands)

    def set_wing_position(self, positions):
        # positions (4x1) vector of position commands for the wings
        commands = {}
        commands['time_stamp'] = np.array([[time.time()]])
        commands['commands'] = positions

        MM.WING_COMMANDS.set(commands)


    def pause_motor_thread(self):
        commands = {'pause':np.array([[1.0]])}
        MM.JOINT_COMMANDS.set(commands)

    def init_sim(self):
        if self.simulation and self.simulator_name is 'Vrep':
            self.simulator = VrepInterface()
            self.simulator.set_dt(self.sim_dt)
            self._parse_joints()
            self.sim_imu = self.simulator.get_object_handle(
                'IMU')  # get the joint handle for the IMU, IMU must be named the same

            self.simulator.start()
            self.simulator.take_step()
        elif self.simulation and self.simulator_name is 'Gazebo':

            # TODO: get rid of number of joint magic number
            self.simulator = ALPHREDGazeboInterface(self.name, 12)
            
            self.simulator.reset_simulation()
            self.simulator.set_step_size(self.sim_dt)

            # TODO: get rid of number of limbs magic number
            p_gains = [GAZEBO_HIP_YAW_P, GAZEBO_HIP_PITCH_P, GAZEBO_KNEE_PITCH_P]*4
            i_gains = [GAZEBO_HIP_YAW_I, GAZEBO_HIP_PITCH_I, GAZEBO_KNEE_PITCH_I]*4
            d_gains = [GAZEBO_HIP_YAW_D, GAZEBO_HIP_PITCH_D, GAZEBO_KNEE_PITCH_D]*4

            self.simulator.set_all_position_pid_gains(p_gains, i_gains, d_gains)

            lower_limits = [HIP_YAW_LIMIT_MINUS, HIP_PITCH_LIMIT_MINUS, KNEE_PITCH_LIMIT_MINUS]*4
            upper_limits = [HIP_YAW_LIMIT_PLUS, HIP_PITCH_LIMIT_PLUS, KNEE_PITCH_LIMIT_PLUS]*4
            self.simulator.set_joint_limits(lower_limits, upper_limits)

            effort_limits = [TORQUE_MAX]*12
            self.simulator.set_effort_limits(effort_limits)

        elif self.simulation and self.simulator_name == None:
            print("No simulator selected.")

        else:
            raise SimulationConfigOffError("Please double check whether simulation is turned on.")

    def stop(self):
        commands = {}
        commands['stop'] = np.array([[1.0]])
        MM.JOINT_COMMANDS.set(commands)

    def _parse_joints(self):

        joint_handles = self.simulator.get_joint_handles()
        for k in self.joint:
            joint_name = self.joint[k]['name']
            if joint_name in joint_handles:
                self.joint[k]['sim_handle'] = joint_handles[joint_name]

    def update_pos(self):

        data = MM.JOINT_STATE.get()

        q = data['joint_positions']
        dq = data['joint_velocities']

        self.joint[LIMB1_HIP_YAW]['q'] = q[0, 0]
        self.joint[LIMB1_HIP_PITCH]['q'] = q[1, 0]
        self.joint[LIMB1_KNEE_PITCH]['q'] = q[2, 0]
        self.joint[LIMB2_HIP_YAW]['q'] = q[3, 0]
        self.joint[LIMB2_HIP_PITCH]['q'] = q[4, 0]
        self.joint[LIMB2_KNEE_PITCH]['q'] = q[5, 0]
        self.joint[LIMB3_HIP_YAW]['q'] = q[6, 0]
        self.joint[LIMB3_HIP_PITCH]['q'] = q[7, 0]
        self.joint[LIMB3_KNEE_PITCH]['q'] = q[8, 0]
        self.joint[LIMB4_HIP_YAW]['q'] = q[9, 0]
        self.joint[LIMB4_HIP_PITCH]['q'] = q[10, 0]
        self.joint[LIMB4_KNEE_PITCH]['q'] = q[11, 0]

        self.joint[LIMB1_HIP_YAW]['dq'] = dq[0, 0]
        self.joint[LIMB1_HIP_PITCH]['dq'] = dq[1, 0]
        self.joint[LIMB1_KNEE_PITCH]['dq'] = dq[2, 0]
        self.joint[LIMB2_HIP_YAW]['dq'] = dq[3, 0]
        self.joint[LIMB2_HIP_PITCH]['dq'] = dq[4, 0]
        self.joint[LIMB2_KNEE_PITCH]['dq'] = dq[5, 0]
        self.joint[LIMB3_HIP_YAW]['dq'] = dq[6, 0]
        self.joint[LIMB3_HIP_PITCH]['dq'] = dq[7, 0]
        self.joint[LIMB3_KNEE_PITCH]['dq'] = dq[8, 0]
        self.joint[LIMB4_HIP_YAW]['dq'] = dq[9, 0]
        self.joint[LIMB4_HIP_PITCH]['dq'] = dq[10, 0]
        self.joint[LIMB4_KNEE_PITCH]['dq'] = dq[11, 0]

        self.FK()

        limb_data ={}
        limb_data['time_stamp'] = np.array([[time.time()]])
        limb_data['mpc'] = np.array([[self.mpc]])
        limb_data['limb1_pos'] = self.joint[LIMB1_TOE]['p']
        limb_data['limb2_pos'] = self.joint[LIMB2_TOE]['p']
        limb_data['limb3_pos'] = self.joint[LIMB3_TOE]['p']
        limb_data['limb4_pos'] = self.joint[LIMB4_TOE]['p']

        MM.LIMB_STATE.set(limb_data)

    def restart_estimator(self):
        print("Restarting Estimator...")
        est_state = MM.ESTIMATOR_STATE.get()
        # Estimator returns 1.0 if it has successfully restarted, or 0.0 otherwise
        est_cmd = {}
        while est_state['est_return_status'] == 0.0:
            est_cmd['restart'] = np.array([[1.0]])
            MM.ESTIMATOR_COMMANDS.set(est_cmd)
        est_cmd['restart'] = np.array([[0.0]])
        MM.ESTIMATOR_COMMANDS.set(est_cmd)


    def print_ee_positions(self):
        print("Limb1:")
        print(self.joint[LIMB1_TOE]['p'])
        print("")
        print("Limb2:")
        print(self.joint[LIMB2_TOE]['p'])
        print("")
        print("Limb3:")
        print(self.joint[LIMB3_TOE]['p'])
        print("")
        print("Limb4:")
        print(self.joint[LIMB4_TOE]['p'])

    def print_joint_positions(self):
        print("Limb1:")
        print("1 - Hip yaw:    %s" % (self.joint[LIMB1_HIP_YAW]['q']))
        print("2 - Hip pitch:  %s" % (self.joint[LIMB1_HIP_PITCH]['q']))
        print("3 - Knee pitch: %s" % (self.joint[LIMB1_KNEE_PITCH]['q']))
        print("")
        print("Limb2:")
        print("4 - Hip yaw:    %s" % (self.joint[LIMB2_HIP_YAW]['q']))
        print("5 - Hip pitch:  %s" % (self.joint[LIMB2_HIP_PITCH]['q']))
        print("6 - Knee pitch: %s" % (self.joint[LIMB2_KNEE_PITCH]['q']))
        print("")
        print("Limb3:")
        print("7 - Hip yaw:    %s" % (self.joint[LIMB3_HIP_YAW]['q']))
        print("8 - Hip pitch:  %s" % (self.joint[LIMB3_HIP_PITCH]['q']))
        print("9 - Knee pitch: %s" % (self.joint[LIMB3_KNEE_PITCH]['q']))
        print("")
        print("Limb4:")
        print("10 - Hip yaw:    %s" % (self.joint[LIMB4_HIP_YAW]['q']))
        print("11 - Hip pitch:  %s" % (self.joint[LIMB4_HIP_PITCH]['q']))
        print("12 - Knee pitch: %s" % (self.joint[LIMB4_KNEE_PITCH]['q']))
        print("")

    def get_time(self):
        if self.simulation and self.simulator_name is 'Vrep':
            return self.simulator.get_last_cmd_time() / 1000.0
        elif self.simulation and self.simulator_name is 'Gazebo':
            return self.simulator.get_current_time()
        else:
            return time.time()

    def clear_save_data(self):
        self.save_data = np.empty([10000, 25])
        self.data_index = 0

    def export_data_csv(self):
        np.savetxt('test_data.csv', self.save_data)


# Exceptions
class JointAxisError(Exception):
    '''When the parameter for the joint axis is not correct (it is not a unit vector towards an axis'''
    pass


class JointRelativePositionError(Exception):
    '''If the shape of the position numpy array is not (3,1)'''
    pass


class SimulationConfigOffError(Exception):
    '''If the simulation settings are not configured correctly'''
    pass


class SimulationReadError(Exception):
    '''If an attempted read from the simulation is not successful'''
    pass


class JointLimitError(Exception):
    '''When the joint is commanded to travel past its limit'''
    pass


class IKError(Exception):
    '''When the desired end-effector position is greater than the length of the two links.'''
    pass
