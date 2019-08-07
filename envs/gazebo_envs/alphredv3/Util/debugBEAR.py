#!usr/bin/env python
from __future__ import division
from __future__ import print_function
from past.utils import old_div
__author__ = "Jeffrey Yu"
__email__ = "c.jeffyu@gmail.com"
__copyright__ = "Copyright 2018 RoMeLa"
__date__ = "May 03, 2018"

__version__ = "0.0.1"
__status__ = "Prototype"

import os
import pdb
import time
import numpy as np
from pybear import Manager

pdm1 = Manager.BEAR(port='/dev/ttyUSB0', baudrate=8000000)
pdm2 = Manager.BEAR(port='/dev/ttyUSB1', baudrate=8000000)
# pdm = Manager.BEAR(port='/dev/ttyUSB0', baudrate=1000000)

# pdm.set_mode((1,0), (2,1), (3,1))
# pdm.set_p_gain_id((1,0), (2,1), (3,1))

pdb.set_trace()

# pdm1.set_torque_enable((1,0),(2,0),(3,0))
# pdm2.set_torque_enable((4,0),(5,0),(6,0))

idx = 0
while True:
    idx += 1
    print("\n{} Loop Start!".format(idx))
    # time.sleep(0.2)
    # print("{}: \t{}".format(time.time(), pdm.get_id(2,3)))
    # print("{}: \t{}".format(time.time(), pdm.get_mode(2,3)))
    # print("{}: \t{}".format(time.time(), pdm.get_torque_enable(2,3)))
    # print("{}: \t{}".format(time.time(), pdm.get_baudrate(2,3)))
    t0 = time.time()
    print("{}: \t{}".format(time.time(), pdm1.get_present_position(1,2,3)))
    print("{}: \t{}".format(time.time(), pdm2.get_present_position(4,5,6)))

    # print("{}".format(pdm1.get_bulk_status((1, 'present_position', 'present_velocity'), 
    #                                        (2, 'present_position', 'present_velocity'),
    #                                        (3, 'present_position', 'present_velocity') )))
    # print("{}".format(pdm2.get_bulk_status((4, 'present_position', 'present_velocity'), 
    #                                        (5, 'present_position', 'present_velocity'),
    #                                        (6, 'present_position', 'present_velocity') )))
    hz = old_div(1,(time.time()-t0))
    print("Freq: {}".format(hz))
