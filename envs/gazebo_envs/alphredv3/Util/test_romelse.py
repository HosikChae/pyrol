#!usr/bin/env python
from __future__ import print_function
__author__      = "Jeff Yu"
__email__       = "c.jeffyu@gmail.com"
__copyright__   = "Copyright 2019 RoMeLa"
__date__        = "April 16, 2019"

__version__     = "0.0.1"
__status__      = "Prototype"

"""
Test the RomeLSE on ALPHRED 
"""

import time
import numpy as np
import Util.MemoryManager as MM
import pdb

MM.connect()

def main():
    time_sleep = 0.5
    t0 = time.time()
    rot_mat = np.zeros((3,3))
    # pdb.set_trace()
    while(1):
        if (time.time() - t0 > time_sleep):
            est_data = MM.ESTIMATOR_STATE.get()
            rot_mat[0,0] = est_data['rot_matrix'][0,0]
            rot_mat[1,0] = est_data['rot_matrix'][1,0]
            rot_mat[2,0] = est_data['rot_matrix'][2,0]
            rot_mat[0,1] = est_data['rot_matrix'][3,0]
            rot_mat[1,1] = est_data['rot_matrix'][4,0]
            rot_mat[2,1] = est_data['rot_matrix'][5,0]
            rot_mat[0,2] = est_data['rot_matrix'][6,0]
            rot_mat[1,2] = est_data['rot_matrix'][7,0]
            rot_mat[2,2] = est_data['rot_matrix'][8,0]
            print("time_stamp: %s" %(est_data['time_stamp']))
            print("position: %s" %(est_data['position']))
            print("velocity: %s" %(est_data['velocity']))
            print("euler_ang: %s" %(est_data['euler_ang']))
            print("rot_matrix: %s" %(est_data['rot_matrix']))
            print("ang_rate: %s" %(est_data['ang_rate']))
            print("rot_mat 3x3:\n%s" %(rot_mat))
            t0 = time.time()


if __name__ == "__main__":
    main()
