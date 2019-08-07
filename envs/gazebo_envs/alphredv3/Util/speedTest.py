#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from past.utils import old_div
__author__  	= "Jeffrey Yu"
__email__   	= "c.jeffyu@gmail.com"
__copyright__ 	= "Copyright 2017 RoMeLa"
__date__ 		= "May 04, 2018"

__version__ 	= "0.1.0"
__status__ 		= "Prototype"

import pdb
import time
import os
import Settings.Robot as NABiV2
from Settings.MACROS_NABiV2 import *

robot = NABiV2.Body

robot.start_drivers()
robot.running_imu = False

end_time = 2.0
curr_time = 0

t0 = time.time()
last_time = t0
i = 0
N = 100

test = 1
# if robot.running_imu:
# 	robot.imu0.node.resume()
# try:
if test == 1:
	while curr_time < end_time:
		robot.update_pos()
		robot.switch_torque(0)
		if (i%N == 0):
			t_elapsed = time.time() - last_time
			last_time = time.time()
			print("Time elapsed for 100 cycles:\t%s" %(t_elapsed))
			print("Average freq: \t%s" %(old_div(N,t_elapsed)))
		i = i + 1
# except:
# 	robot.export_data_csv()
	
if test == 2:
	while curr_time < end_time:
		robot.update_pos()
		robot.switch_torque(0)
		while (time.time() - last_time < robot.dt):
			pass
		print("Elapsed time: %s" %(time.time() - last_time))
		last_time = time.time()
		curr_time = time.time() - t0


# if robot.running_imu:
# 	robot.imu0.node.setToIdle()
