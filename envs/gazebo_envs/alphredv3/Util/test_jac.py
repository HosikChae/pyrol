#!usr/bin/env python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from past.utils import old_div
__author__      = "Jeff Yu"
__email__       = "c.jeffyu@gmail.com"
__copyright__   = "Copyright 2019 RoMeLa"
__date__        = "April 9, 2017"

__version__     = "1.0.0"
__status__      = "Prototype"

import numpy as np
from . import Dynamics_and_Kinematics as dk
import time
import pdb

q = np.zeros((12,1))
dq = np.ones((12,1))
F = np.ones((12,1))
# q = np.random.rand(12,1)
# dq = np.random.rand(12,1)
# F = np.random.rand(12,1)

J = dk.jacobian_all_limb(q)

v = dk.jacobian_velocity(q,dq)

tau = dk.jacobian_torque(q,F)

print("Continue to run speed test")
pdb.set_trace()

n_run = 1000
t0 = time.time()
for i in range(n_run):
    # J = dk.jacobian_all_limb(q)
    v = dk.jacobian_velocity(q,dq)
    tau = dk.jacobian_torque(q,F)

avg_time = old_div((time.time() - t0),n_run)
print("avg_time: %s" %(avg_time)) 