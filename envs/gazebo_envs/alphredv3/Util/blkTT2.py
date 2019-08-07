#!usr/bin/env python
from __future__ import print_function
from builtins import range
__author__ = "Min Sung Ahn"
__email__ = "aminsung@gmail.com"
__copyright__ = "Copyright 2018 RoMeLa"
__date__ = "July 13, 2018"

__version__ = "0.0.1"
__status__ = "Prototype"


import pdb
import time

import numpy as np

import Settings.Robot as Robot
from pybear import Manager

def run_test():
    pdm = Manager.BEAR(port=Robot.pdm_port2, baudrate=Robot.baudrate)
    pdm.set_limit_id_max((4, 0.0),(5, 0.0),(6, 0.0))
    pdm.set_limit_iq_max((4, Robot.max_iq),(5, Robot.max_iq),(6, Robot.max_iq))
    pdm.set_limit_position_max((4, 32768.0),(5, 76000.0),(6, 87381))
    pdm.set_limit_position_min((4,-32768.0),(5,  5000.0),(6,-87381))
    pdm.set_p_gain_position((4, Robot.hip_yaw_pkp),(5, Robot.hip_pitch_pkp),(6, Robot.knee_pitch_pkp))
    pdm.set_i_gain_position((4, Robot.hip_yaw_pki),(5, Robot.hip_pitch_pki),(6, Robot.knee_pitch_pki))
    pdm.set_d_gain_position((4, Robot.hip_yaw_pkd),(5, Robot.hip_pitch_pkd),(6, Robot.knee_pitch_pkd))
    pdm.set_p_gain_velocity((4, Robot.hip_yaw_vkp),(5, Robot.hip_pitch_vkp),(6, Robot.knee_pitch_vkp))
    pdm.set_i_gain_velocity((4, Robot.hip_yaw_vki),(5, Robot.hip_pitch_vki),(6, Robot.knee_pitch_vki))
    pdm.set_d_gain_velocity((4, Robot.hip_yaw_vkd),(5, Robot.hip_pitch_vkd),(6, Robot.knee_pitch_vkd))
    pdm.set_p_gain_force((4, Robot.hip_yaw_fkp),(5, Robot.hip_pitch_fkp),(6, Robot.knee_pitch_fkp))
    pdm.set_i_gain_force((4, Robot.hip_yaw_fki),(5, Robot.hip_pitch_fki),(6, Robot.knee_pitch_fki))
    pdm.set_d_gain_force((4, Robot.hip_yaw_fkd),(5, Robot.hip_pitch_fkd),(6, Robot.knee_pitch_fkd))

    pdm.set_torque_enable((4, 0),(5, 0),(6, 0))
    mode = 0
    pdm.set_mode((4, mode),(5, mode),(6, mode))
    pdm.set_torque_enable((4, 1),(5, 1),(6, 1))

    print("Press (c) to start timing")
    pdb.set_trace()
    dpt = []
    for _ in range(10000):
        t0 = time.time()
        pdm.set_goal_position((4, 0), (5, 0), (6, 0))

        pdm.get_bulk_status((4, 'present_position', 'present_velocity', 'present_iq'), (5, 'present_position', 'present_velocity', 'present_iq'), (6, 'present_position', 'present_velocity', 'present_iq'))
        tf = time.time()
        tlap = tf-t0
        print("Hz: {}".format(1./tlap))
        dpt.append(tlap)
    
    dpt = np.asarray(dpt)
    np.savetxt("l2_op.csv", dpt, delimiter=",")

if __name__ == "__main__":
    run_test()
    print("Done")
