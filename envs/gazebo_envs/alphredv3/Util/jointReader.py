#!usr/bin/env python
from __future__ import division
from builtins import range
from past.utils import old_div
__author__ = "Joshua Hooks"
__email__ = "hooksjrose@gmail.com"
__copyright__ = "Copyright 2019 RoMeLa"
__date__ = "March 4th, 2019"

__version__ = "0.0.1"
__status__ = "Prototype"

import time
import os
import Settings.Robot as ALPHRED
import pdb
import numpy as np
import pdb
import gc


Robot = ALPHRED.Body

t0 = time.time()
last_time = t0

dt = 0.002
TF = 600.0

n = int(old_div(TF,dt))

data = np.zeros((n,1))
#datagc = np.zeros((n,1))

for i in range(n):
	data[i,0] = time.time() - t0
    #    gccount = gc.get_count()
    #    datagc[i, 0] = gccount[0]
	t0 = time.time()
	Robot.update_pos()
	Robot.set_command_position()



np.savetxt('timing_data.csv', data)
#np.savetxt('datagc.csv',datagc)

"""
while 1:
    # t0 = time.time()
	os.system('clear')
	#print time.time() - t0
	#t0 = time.time()
	Robot.update_pos()
	#Robot.set_command_position()
	Robot.print_joint_positions()
        time.sleep(0.3)
"""

"""
	print "==== Body Pitch and Rates ===="
	print "robot.joint[BODY]['q']: %s" %(robot.joint[BODY]['q'])
	print "robot.joint[BODY]['dq']: %s" %(robot.joint[BODY]['dq'])

	print "==== LEG 1 Angles ===="
	print "robot.joint[LEG1_HIP_YAW]['q']: %s" %(robot.joint[LEG1_HIP_YAW]['q'])	
	print "robot.joint[LEG1_HIP_PITCH]['q']: %s" %(robot.joint[LEG1_HIP_PITCH]['q'])
	print "robot.joint[LEG1_KNEE_PITCH]['q']: %s" %(robot.joint[LEG1_KNEE_PITCH]['q'])
	# print "robot.joint[LEG1_TOE]['p']:\n%s" %(robot.joint[LEG1_TOE]['p'])
	print "==== LEG 2 Angles ===="
	print "robot.joint[LEG2_HIP_YAW]['q']: %s" %(robot.joint[LEG2_HIP_YAW]['q'])	
	print "robot.joint[LEG2_HIP_PITCH]['q']: %s" %(robot.joint[LEG2_HIP_PITCH]['q'])
	print "robot.joint[LEG2_KNEE_PITCH]['q']: %s" %(robot.joint[LEG2_KNEE_PITCH]['q'])
	# print "robot.joint[LEG2_TOE]['p']:\n%s" %(robot.joint[LEG2_TOE]['p'])

	print "==== LEG 1 Rates ===="
	print "robot.joint[LEG1_HIP_YAW]['dq']: %s" %(robot.joint[LEG1_HIP_YAW]['dq'])	
	print "robot.joint[LEG1_HIP_PITCH]['dq']: %s" %(robot.joint[LEG1_HIP_PITCH]['dq'])
	print "robot.joint[LEG1_KNEE_PITCH]['dq']: %s" %(robot.joint[LEG1_KNEE_PITCH]['dq'])
	print "==== LEG 2 Rates ===="
	print "robot.joint[LEG2_HIP_YAW]['dq']: %s" %(robot.joint[LEG2_HIP_YAW]['dq'])	
	print "robot.joint[LEG2_HIP_PITCH]['dq']: %s" %(robot.joint[LEG2_HIP_PITCH]['dq'])
	print "robot.joint[LEG2_KNEE_PITCH]['dq']: %s" %(robot.joint[LEG2_KNEE_PITCH]['dq'])

	print "robot.joint[LEG1_TOE]['p']:\n%s" %(robot.joint[LEG1_TOE]['p'])
	print "robot.joint[LEG2_TOE]['p']:\n%s" %(robot.joint[LEG2_TOE]['p'])
"""


