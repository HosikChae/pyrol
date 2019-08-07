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
    pdm = Manager.BEAR(port=Robot.pdm_port1, baudrate=Robot.baudrate)
    pdm.set_limit_id_max((1, 0.0),(2, 0.0),(3, 0.0))
    pdm.set_limit_iq_max((1, Robot.max_iq),(2, Robot.max_iq),(3, Robot.max_iq))
    pdm.set_limit_position_max((1, 32768.0),(2, 76000.0),(3, 87381))
    pdm.set_limit_position_min((1,-32768.0),(2,  5000.0),(3,-87381))
    pdm.set_p_gain_position((1, Robot.hip_yaw_pkp),(2, Robot.hip_pitch_pkp),(3, Robot.knee_pitch_pkp))
    pdm.set_i_gain_position((1, Robot.hip_yaw_pki),(2, Robot.hip_pitch_pki),(3, Robot.knee_pitch_pki))
    pdm.set_d_gain_position((1, Robot.hip_yaw_pkd),(2, Robot.hip_pitch_pkd),(3, Robot.knee_pitch_pkd))
    pdm.set_p_gain_velocity((1, Robot.hip_yaw_vkp),(2, Robot.hip_pitch_vkp),(3, Robot.knee_pitch_vkp))
    pdm.set_i_gain_velocity((1, Robot.hip_yaw_vki),(2, Robot.hip_pitch_vki),(3, Robot.knee_pitch_vki))
    pdm.set_d_gain_velocity((1, Robot.hip_yaw_vkd),(2, Robot.hip_pitch_vkd),(3, Robot.knee_pitch_vkd))
    pdm.set_p_gain_force((1, Robot.hip_yaw_fkp),(2, Robot.hip_pitch_fkp),(3, Robot.knee_pitch_fkp))
    pdm.set_i_gain_force((1, Robot.hip_yaw_fki),(2, Robot.hip_pitch_fki),(3, Robot.knee_pitch_fki))
    pdm.set_d_gain_force((1, Robot.hip_yaw_fkd),(2, Robot.hip_pitch_fkd),(3, Robot.knee_pitch_fkd))

    pdm.set_torque_enable((1, 0),(2, 0),(3, 0))
    mode = 0
    pdm.set_mode((1, mode),(2, mode),(3, mode))
    pdm.set_torque_enable((1, 1),(2, 1),(3, 1))

    print("Press (c) to start timing")
    pdb.set_trace()
    dpt = []
    for _ in range(10000):
        t0 = time.time()
        pdm.set_goal_position((1, 0), (2, 0), (3, 0))

        pdm.get_bulk_status((1, 'present_position', 'present_velocity', 'present_iq'), (2, 'present_position', 'present_velocity', 'present_iq'), (3, 'present_position', 'present_velocity', 'present_iq'))
        tf = time.time()
        tlap = tf-t0
        print("Hz: {}".format(1./tlap))
        dpt.append(tlap)
    
    dpt = np.asarray(dpt)
    np.savetxt("l1_op.csv", dpt, delimiter=",")

if __name__ == "__main__":
    run_test()
    print("Done")
