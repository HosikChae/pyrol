from gazebopy import RobotInterface
import numpy as np
import pyshmxtreme.SHMSegment as shmx
import posix_ipc

"""
This class is specific to ALPHREDV3 to interface with Gazebo, which inherits from a general class that interfaces
with Gazebo
"""

class ALPHREDGazeboInterface(RobotInterface):
    def __init__(self, robot_name, num_joints):
        RobotInterface.__init__(self, robot_name, num_joints)

        self._initialize_alphred_shared_memory()


    def _initialize_alphred_shared_memory(self):
        self._limb1_contact = shmx.SHMSegment(robot_name=self.robot_name, seg_name='LIMB1_CONTACT', init=False)
        self._limb1_contact.add_blocks(name='time_stamp', data=np.zeros((1, 1)))
        self._limb1_contact.add_blocks(name='on', data=np.zeros((1, 1)))

        self._limb2_contact = shmx.SHMSegment(robot_name=self.robot_name, seg_name='LIMB2_CONTACT', init=False)
        self._limb2_contact.add_blocks(name='time_stamp', data=np.zeros((1, 1)))
        self._limb2_contact.add_blocks(name='on', data=np.zeros((1, 1)))

        self._limb3_contact = shmx.SHMSegment(robot_name=self.robot_name, seg_name='LIMB3_CONTACT', init=False)
        self._limb3_contact.add_blocks(name='time_stamp', data=np.zeros((1, 1)))
        self._limb3_contact.add_blocks(name='on', data=np.zeros((1, 1)))

        self._limb4_contact = shmx.SHMSegment(robot_name=self.robot_name, seg_name='LIMB4_CONTACT', init=False)
        self._limb4_contact.add_blocks(name='time_stamp', data=np.zeros((1, 1)))
        self._limb4_contact.add_blocks(name='on', data=np.zeros((1, 1)))

        try:
            self._limb1_contact.connect_segment()
            self._limb2_contact.connect_segment()
            self._limb3_contact.connect_segment()
            self._limb4_contact.connect_segment()

        except posix_ipc.ExistentialError as error:
            self._limb1_contact.initialize = True
            self._limb2_contact.initialize = True
            self._limb3_contact.initialize = True
            self._limb4_contact.initialize = True

            self._limb1_contact.connect_segment()
            self._limb2_contact.connect_segment()
            self._limb3_contact.connect_segment()
            self._limb4_contact.connect_segment()

    def get_foot_contacts(self):
        return np.array([[self._limb1_contact.get()['on'][0, 0]], [self._limb2_contact.get()['on'][0, 0]],
                         [self._limb3_contact.get()['on'][0, 0]], [self._limb4_contact.get()['on'][0, 0]]])