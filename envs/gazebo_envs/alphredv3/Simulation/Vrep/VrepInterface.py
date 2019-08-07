#!usr/bin/env python
from builtins import range
__author__ = "Hosik Chae"
__email__ = "CKMagenta@gmail.com"
__copyright__ = "Copyright 2016 RoMeLa"

__version__ = "0.0.1"
__status__ = "Prototype"

'''
  This class is used to handle the communication between python and V-REP
'''

import numpy as np
# TODO: define `VREP_LIB_DIR` in ~/.bashrc, or run `sys.path.append(VREP_LIB_DIR)`
from vrepy import BasicInterface
from vrepy import RobotInterface

class ALPHREDInterface(RobotInterface):
    def __init__(self):
        super(ALPHREDInterface, self).__init__()
        self._robot_interface = super(ALPHREDInterface, self)
        self.set_dt(50e-3)  # 50 ms

    def set_controller_position(self):
        self.set_control_scheme(self.CONTROL_SCHEME_POSITION_PID)

    # For Backward Compatibility, Temporarily modified to add "batch_comm".
    # def set_limb_command(self, joint, commands, limb_num):
    #     for idx in range(0,3):
    #         k = (limb_num-1)*4 + 1 + idx
    #         super(RobotInterface, self).set_joint_target_position(joint[k]['sim_handle'], commands[idx])
    def set_limb_command(self, joint, commands, limb_num, batch_comm=True):
        for idx in range(0, 3):
            k = (limb_num - 1) * 4 + 1 + idx
            RobotInterface.set_joint_target_position(self, joint[k], commands[idx], batch_comm)

    def set_limb1_command(self, joint, commands):
        self.set_limb_command(joint, commands, 1)

    def set_limb2_command(self, joint, commands):
        self.set_limb_command(joint, commands, 2)

    def set_limb3_command(self, joint, commands):
        self.set_limb_command(joint, commands, 3)

    def set_limb4_command(self, joint, commands):
        self.set_limb_command(joint, commands, 4)


    # Alias
    def get_current_torque(self, joint):
        # Named after the API name "getJointForce", not torque but force
        return self.get_current_force(joint)

    def send_command(self):
        self.take_step()


# For Backward Compatibility
class VrepInterface(ALPHREDInterface):
    def __init__(self):
        super(VrepInterface, self).__init__()

    def set_dt(self, dt):
        # Defined in BasicInterface
        ALPHREDInterface.set_dt(self,dt)

    def start(self):
        # Defined in BasicInterface
        ALPHREDInterface.start(self)

    def stop(self):
        # Defined in BasicInterface
        ALPHREDInterface.stop(self)

    def get_joint_handles(self):
        # This feature is automatically done in RobotInterface.__init__()
        return self._sim_joint_handles

    def get_current_position(self, joint):
        return ALPHREDInterface.get_current_position(self, joint)

    def get_current_torque(self, joint):
        # Named after the API name "getJointForce", not torque but force
        return ALPHREDInterface.get_current_force(self, joint)

    def set_command_position(self, joint, commands):
        _opmode = self.opmode_setter
        self.opmode_setter = vrep.simx_opmode_blocking
        self.set_joint_target_position(joint, commands, batch_comm=False)
        self.opmode_setter = _opmode
        self.take_step()

    def set_limb_command(self, joint, commands, limb_num):
        _opmode = self.opmode_setter
        self.opmode_setter = vrep.simx_opmode_blocking
        ALPHREDInterface.set_limb_command(self, joint, commands, limb_num, batch_comm=False)
        self.opmode_setter = _opmode

    def send_command(self):
        self.take_step()

    def set_limb1_command(self, joint, commands):
        self.set_limb_command(joint, commands, 1)

    def set_limb2_command(self, joint, commands):
        self.set_limb_command(joint, commands, 2)

    def set_limb3_command(self, joint, commands):
        self.set_limb_command(joint, commands, 3)

    def set_limb4_command(self, joint, commands):
        self.set_limb_command(joint, commands, 4)
