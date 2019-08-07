from __future__ import print_function
from builtins import range
import Settings.Robot as ALPHRED
import time

robot = ALPHRED.Body

robot.IK()

robot.old_IK()

t0 = time.time()
for i in range(1,10000):
    robot.IK()

time1 = time.time() - t0
print(time1)

t0 = time.time()
for i in range(1,10000):
    robot.old_IK()

time2 = time.time() - t0
print(time2)
