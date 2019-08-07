#!usr/bin/env python
from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
__author__ = "Jeffrey Yu"
__email__ = "c.jeffyu@gmail.com"
__copyright__ = "Copyright 2017 RoMeLa"
__date__ = "April 25, 2018"

__version__ = "0.0.1"
__status__ = "Prototype"

import pdb
import time
import numpy as np
from Settings.MACROS_NABiV2 import *

#### Inertial parameters
(m0,m1,m2,m3,m4) = MASS_BODY, MASS_FEMUR, MASS_TIBIA, MASS_FEMUR, MASS_TIBIA
(I0xx,I0xy,I0yy,I0yz,I0zz,I0zx) = INERTIA_BODY_XX,INERTIA_BODY_XY,INERTIA_BODY_YY,INERTIA_BODY_YZ,INERTIA_BODY_ZZ,INERTIA_BODY_ZX 
(I1xx,I1xy,I1yy,I1yz,I1zz,I1zx) = INERTIA_FEMUR_XX,INERTIA_FEMUR_XY,INERTIA_FEMUR_YY,INERTIA_FEMUR_YZ,INERTIA_FEMUR_ZZ,INERTIA_FEMUR_ZX 
(I2xx,I2xy,I2yy,I2yz,I2zz,I2zx) = INERTIA_TIBIA_XX,INERTIA_TIBIA_XY,INERTIA_TIBIA_YY,INERTIA_TIBIA_YZ,INERTIA_TIBIA_ZZ,INERTIA_TIBIA_ZX 
(I3xx,I3xy,I3yy,I3yz,I3zz,I3zx) = INERTIA_FEMUR_XX,INERTIA_FEMUR_XY,INERTIA_FEMUR_YY,INERTIA_FEMUR_YZ,INERTIA_FEMUR_ZZ,INERTIA_FEMUR_ZX 
(I4xx,I4xy,I4yy,I4yz,I4zz,I4zx) = INERTIA_TIBIA_XX,INERTIA_TIBIA_XY,INERTIA_TIBIA_YY,INERTIA_TIBIA_YZ,INERTIA_TIBIA_ZZ,INERTIA_TIBIA_ZX 
#### Length Properties - can be negative
(L0x,L0z) = BODY_OFFSET_X, -ROTOR_OFFSET_Z
(L1x,L1z) = LENGTH_FEMUR_X, -LENGTH_FEMUR_Z
(L2x,L2z) = LENGTH_TIBIA_X, -LENGTH_TIBIA_Z
(L3x,L3z) = LENGTH_FEMUR_X, -LENGTH_FEMUR_Z
(L4x,L4z) = LENGTH_TIBIA_X, -LENGTH_TIBIA_Z
(c0x,c0z) = LC_BODY_X, LC_BODY_Z
(c1x,c1z) = -LC_FEMUR_X, -LC_FEMUR_Z
(c2x,c2z) = -LC_TIBIA_X, -LC_TIBIA_Z
(c3x,c3z) = -LC_FEMUR_X, -LC_FEMUR_Z
(c4x,c4z) = -LC_TIBIA_X, -LC_TIBIA_Z
#### Gravity
g = GRAV

def main():
	Nruns = 1
	start1 = time.time()
	for i in range(Nruns):
		qddX = np.random.rand(17)
		tau = invdynDS(*qddX)
	end1 = time.time()
	print("Total time for all runs:\t%s" %(end1-start1))
	print("Average time per run:\t%s" %(old_div((end1-start1),Nruns)))


def invdynDS(xh,zh,vh,q1,q2,q3,q4, dxh,dzh,dvh,dq1,dq2,dq3,dq4, ddx,ddz,ddvh):
	#### Construct the matrices needed:
	# Inertia matrix
	H = np.zeros((7,7))
	H[0,0] = m0 + m1 + m2 + m3 + m4
	H[0,2] = L0x*m3*np.sin(vh) - L0x*m2*np.sin(vh) - L0x*m1*np.sin(vh) + L0x*m4*np.sin(vh) + c0z*m0*np.cos(vh) - c0x*m0*np.sin(vh) + c2z*m2*np.cos(q1 + q2 + vh) + c4z*m4*np.cos(q3 + q4 + vh) - c2x*m2*np.sin(q1 + q2 + vh) - c4x*m4*np.sin(q3 + q4 + vh) + L1z*m2*np.cos(q1 + vh) + L3z*m4*np.cos(q3 + vh) - L1x*m2*np.sin(q1 + vh) - L3x*m4*np.sin(q3 + vh) + c1z*m1*np.cos(q1 + vh) + c3z*m3*np.cos(q3 + vh) - c1x*m1*np.sin(q1 + vh) - c3x*m3*np.sin(q3 + vh) + L0z*m1*np.cos(vh) + L0z*m2*np.cos(vh) + L0z*m3*np.cos(vh) + L0z*m4*np.cos(vh)
	H[0,3] = c2z*m2*np.cos(q1 + q2 + vh) - c2x*m2*np.sin(q1 + q2 + vh) + L1z*m2*np.cos(q1 + vh) - L1x*m2*np.sin(q1 + vh) + c1z*m1*np.cos(q1 + vh) - c1x*m1*np.sin(q1 + vh)
	H[0,4] = c2z*m2*np.cos(q1 + q2 + vh) - c2x*m2*np.sin(q1 + q2 + vh)
	H[0,5] = c4z*m4*np.cos(q3 + q4 + vh) - c4x*m4*np.sin(q3 + q4 + vh) + L3z*m4*np.cos(q3 + vh) - L3x*m4*np.sin(q3 + vh) + c3z*m3*np.cos(q3 + vh) - c3x*m3*np.sin(q3 + vh)
	H[0,6] = c4z*m4*np.cos(q3 + q4 + vh) - c4x*m4*np.sin(q3 + q4 + vh)
	H[1,1] = m0 + m1 + m2 + m3 + m4
	H[1,2] = L0x*m3*np.cos(vh) - L0z*m2*np.sin(vh) - L0z*m3*np.sin(vh) - L0z*m4*np.sin(vh) - c0x*m0*np.cos(vh) - c0z*m0*np.sin(vh) - c2x*m2*np.cos(q1 + q2 + vh) - c4x*m4*np.cos(q3 + q4 + vh) - c2z*m2*np.sin(q1 + q2 + vh) - c4z*m4*np.sin(q3 + q4 + vh) - L1x*m2*np.cos(q1 + vh) - L3x*m4*np.cos(q3 + vh) - L1z*m2*np.sin(q1 + vh) - L3z*m4*np.sin(q3 + vh) - c1x*m1*np.cos(q1 + vh) - c3x*m3*np.cos(q3 + vh) - c1z*m1*np.sin(q1 + vh) - c3z*m3*np.sin(q3 + vh) - L0x*m1*np.cos(vh) - L0x*m2*np.cos(vh) - L0z*m1*np.sin(vh) + L0x*m4*np.cos(vh)
	H[1,3] = - c2x*m2*np.cos(q1 + q2 + vh) - c2z*m2*np.sin(q1 + q2 + vh) - L1x*m2*np.cos(q1 + vh) - L1z*m2*np.sin(q1 + vh) - c1x*m1*np.cos(q1 + vh) - c1z*m1*np.sin(q1 + vh)
	H[1,4] = - c2x*m2*np.cos(q1 + q2 + vh) - c2z*m2*np.sin(q1 + q2 + vh)
	H[1,5] = - c4x*m4*np.cos(q3 + q4 + vh) - c4z*m4*np.sin(q3 + q4 + vh) - L3x*m4*np.cos(q3 + vh) - L3z*m4*np.sin(q3 + vh) - c3x*m3*np.cos(q3 + vh) - c3z*m3*np.sin(q3 + vh)
	H[1,6] = - c4x*m4*np.cos(q3 + q4 + vh) - c4z*m4*np.sin(q3 + q4 + vh)
	H[2,0] = H[0,2]
	H[2,1] = H[1,2]
	H[2,2] = I0yy + I1yy + I2yy + I3yy + I4yy + L0x**2*m1 + L0x**2*m2 + L0x**2*m3 + L1x**2*m2 + L0x**2*m4 + L3x**2*m4 + L0z**2*m1 + L0z**2*m2 + L0z**2*m3 + L1z**2*m2 + L0z**2*m4 + L3z**2*m4 + c0x**2*m0 + c1x**2*m1 + c2x**2*m2 + c3x**2*m3 + c4x**2*m4 + c0z**2*m0 + c1z**2*m1 + c2z**2*m2 + c3z**2*m3 + c4z**2*m4 + 2*L0x*c2x*m2*np.cos(q1 + q2) - 2*L0x*c4x*m4*np.cos(q3 + q4) + 2*L0z*c2z*m2*np.cos(q1 + q2) + 2*L0z*c4z*m4*np.cos(q3 + q4) + 2*L0x*c2z*m2*np.sin(q1 + q2) - 2*L0z*c2x*m2*np.sin(q1 + q2) - 2*L0x*c4z*m4*np.sin(q3 + q4) - 2*L0z*c4x*m4*np.sin(q3 + q4) + 2*L0x*L1x*m2*np.cos(q1) - 2*L0x*L3x*m4*np.cos(q3) + 2*L0z*L1z*m2*np.cos(q1) + 2*L0z*L3z*m4*np.cos(q3) + 2*L0x*L1z*m2*np.sin(q1) - 2*L1x*L0z*m2*np.sin(q1) - 2*L0x*L3z*m4*np.sin(q3) - 2*L3x*L0z*m4*np.sin(q3) + 2*L0x*c1x*m1*np.cos(q1) + 2*L1x*c2x*m2*np.cos(q2) - 2*L0x*c3x*m3*np.cos(q3) + 2*L3x*c4x*m4*np.cos(q4) + 2*L0z*c1z*m1*np.cos(q1) + 2*L1z*c2z*m2*np.cos(q2) + 2*L0z*c3z*m3*np.cos(q3) + 2*L3z*c4z*m4*np.cos(q4) + 2*L0x*c1z*m1*np.sin(q1) - 2*L0z*c1x*m1*np.sin(q1) + 2*L1x*c2z*m2*np.sin(q2) - 2*L1z*c2x*m2*np.sin(q2) - 2*L0x*c3z*m3*np.sin(q3) - 2*L0z*c3x*m3*np.sin(q3) + 2*L3x*c4z*m4*np.sin(q4) - 2*L3z*c4x*m4*np.sin(q4)
	H[2,3] = I1yy + I2yy + L1x**2*m2 + L1z**2*m2 + c1x**2*m1 + c2x**2*m2 + c1z**2*m1 + c2z**2*m2 + L0x*c2x*m2*np.cos(q1 + q2) + L0z*c2z*m2*np.cos(q1 + q2) + L0x*c2z*m2*np.sin(q1 + q2) - L0z*c2x*m2*np.sin(q1 + q2) + L0x*L1x*m2*np.cos(q1) + L0z*L1z*m2*np.cos(q1) + L0x*L1z*m2*np.sin(q1) - L1x*L0z*m2*np.sin(q1) + L0x*c1x*m1*np.cos(q1) + 2*L1x*c2x*m2*np.cos(q2) + L0z*c1z*m1*np.cos(q1) + 2*L1z*c2z*m2*np.cos(q2) + L0x*c1z*m1*np.sin(q1) - L0z*c1x*m1*np.sin(q1) + 2*L1x*c2z*m2*np.sin(q2) - 2*L1z*c2x*m2*np.sin(q2)
	H[2,4] = I2yy + c2x**2*m2 + c2z**2*m2 + L0x*c2x*m2*np.cos(q1 + q2) + L0z*c2z*m2*np.cos(q1 + q2) + L0x*c2z*m2*np.sin(q1 + q2) - L0z*c2x*m2*np.sin(q1 + q2) + L1x*c2x*m2*np.cos(q2) + L1z*c2z*m2*np.cos(q2) + L1x*c2z*m2*np.sin(q2) - L1z*c2x*m2*np.sin(q2)
	H[2,5] = I3yy + I4yy + L3x**2*m4 + L3z**2*m4 + c3x**2*m3 + c4x**2*m4 + c3z**2*m3 + c4z**2*m4 - L0x*c4x*m4*np.cos(q3 + q4) + L0z*c4z*m4*np.cos(q3 + q4) - L0x*c4z*m4*np.sin(q3 + q4) - L0z*c4x*m4*np.sin(q3 + q4) - L0x*L3x*m4*np.cos(q3) + L0z*L3z*m4*np.cos(q3) - L0x*L3z*m4*np.sin(q3) - L3x*L0z*m4*np.sin(q3) - L0x*c3x*m3*np.cos(q3) + 2*L3x*c4x*m4*np.cos(q4) + L0z*c3z*m3*np.cos(q3) + 2*L3z*c4z*m4*np.cos(q4) - L0x*c3z*m3*np.sin(q3) - L0z*c3x*m3*np.sin(q3) + 2*L3x*c4z*m4*np.sin(q4) - 2*L3z*c4x*m4*np.sin(q4)
	H[2,6] = I4yy + c4x**2*m4 + c4z**2*m4 - L0x*c4x*m4*np.cos(q3 + q4) + L0z*c4z*m4*np.cos(q3 + q4) - L0x*c4z*m4*np.sin(q3 + q4) - L0z*c4x*m4*np.sin(q3 + q4) + L3x*c4x*m4*np.cos(q4) + L3z*c4z*m4*np.cos(q4) + L3x*c4z*m4*np.sin(q4) - L3z*c4x*m4*np.sin(q4)
	H[3,0] = H[0,3]
	H[3,1] = H[1,3]
	H[3,2] = H[2,3]
	H[3,3] = m2*L1x**2 + 2*m2*np.cos(q2)*L1x*c2x + 2*m2*np.sin(q2)*L1x*c2z + m2*L1z**2 - 2*m2*np.sin(q2)*L1z*c2x + 2*m2*np.cos(q2)*L1z*c2z + m1*c1x**2 + m2*c2x**2 + m1*c1z**2 + m2*c2z**2 + I1yy + I2yy
	H[3,4] = I2yy + c2x**2*m2 + c2z**2*m2 + L1x*c2x*m2*np.cos(q2) + L1z*c2z*m2*np.cos(q2) + L1x*c2z*m2*np.sin(q2) - L1z*c2x*m2*np.sin(q2)
	H[4,0] = H[0,4]
	H[4,1] = H[1,4]
	H[4,2] = H[2,4]
	H[4,3] = H[3,4]
	H[4,4] = m2*c2x**2 + m2*c2z**2 + I2yy
	H[5,0] = H[0,5]
	H[5,1] = H[1,5]
	H[5,2] = H[2,5]
	H[5,5] = m4*L3x**2 + 2*m4*np.cos(q4)*L3x*c4x + 2*m4*np.sin(q4)*L3x*c4z + m4*L3z**2 - 2*m4*np.sin(q4)*L3z*c4x + 2*m4*np.cos(q4)*L3z*c4z + m3*c3x**2 + m4*c4x**2 + m3*c3z**2 + m4*c4z**2 + I3yy + I4yy
	H[5,6] = I4yy + c4x**2*m4 + c4z**2*m4 + L3x*c4x*m4*np.cos(q4) + L3z*c4z*m4*np.cos(q4) + L3x*c4z*m4*np.sin(q4) - L3z*c4x*m4*np.sin(q4)
	H[6,0] = H[0,6]
	H[6,1] = H[1,6]
	H[6,2] = H[2,6]
	H[6,5] = H[5,6]
	H[6,6] = m4*c4x**2 + m4*c4z**2 + I4yy
	# Gravity and velocity products
	CG = np.zeros((7,1))
	CG[0,0] = L0x*dvh**2*m3*np.cos(vh) - c2x*dq2**2*m2*np.cos(q1 + q2 + vh) - c4x*dq3**2*m4*np.cos(q3 + q4 + vh) - c4x*dq4**2*m4*np.cos(q3 + q4 + vh) - c2x*dvh**2*m2*np.cos(q1 + q2 + vh) - c4x*dvh**2*m4*np.cos(q3 + q4 + vh) - c2z*dq1**2*m2*np.sin(q1 + q2 + vh) - c2z*dq2**2*m2*np.sin(q1 + q2 + vh) - c4z*dq3**2*m4*np.sin(q3 + q4 + vh) - c4z*dq4**2*m4*np.sin(q3 + q4 + vh) - c2z*dvh**2*m2*np.sin(q1 + q2 + vh) - c4z*dvh**2*m4*np.sin(q3 + q4 + vh) - L1x*dq1**2*m2*np.cos(q1 + vh) - L3x*dq3**2*m4*np.cos(q3 + vh) - L1x*dvh**2*m2*np.cos(q1 + vh) - L3x*dvh**2*m4*np.cos(q3 + vh) - L1z*dq1**2*m2*np.sin(q1 + vh) - L3z*dq3**2*m4*np.sin(q3 + vh) - L1z*dvh**2*m2*np.sin(q1 + vh) - L3z*dvh**2*m4*np.sin(q3 + vh) - c1x*dq1**2*m1*np.cos(q1 + vh) - c3x*dq3**2*m3*np.cos(q3 + vh) - c1x*dvh**2*m1*np.cos(q1 + vh) - c3x*dvh**2*m3*np.cos(q3 + vh) - c1z*dq1**2*m1*np.sin(q1 + vh) - c3z*dq3**2*m3*np.sin(q3 + vh) - c1z*dvh**2*m1*np.sin(q1 + vh) - c3z*dvh**2*m3*np.sin(q3 + vh) - L0x*dvh**2*m1*np.cos(vh) - L0x*dvh**2*m2*np.cos(vh) - c2x*dq1**2*m2*np.cos(q1 + q2 + vh) + L0x*dvh**2*m4*np.cos(vh) - L0z*dvh**2*m1*np.sin(vh) - L0z*dvh**2*m2*np.sin(vh) - L0z*dvh**2*m3*np.sin(vh) - L0z*dvh**2*m4*np.sin(vh) - c0x*dvh**2*m0*np.cos(vh) - c0z*dvh**2*m0*np.sin(vh) - 2*c2x*dq1*dq2*m2*np.cos(q1 + q2 + vh) - 2*c4x*dq3*dq4*m4*np.cos(q3 + q4 + vh) - 2*c2x*dq1*dvh*m2*np.cos(q1 + q2 + vh) - 2*c2x*dq2*dvh*m2*np.cos(q1 + q2 + vh) - 2*c4x*dq3*dvh*m4*np.cos(q3 + q4 + vh) - 2*c4x*dq4*dvh*m4*np.cos(q3 + q4 + vh) - 2*c2z*dq1*dq2*m2*np.sin(q1 + q2 + vh) - 2*c4z*dq3*dq4*m4*np.sin(q3 + q4 + vh) - 2*c2z*dq1*dvh*m2*np.sin(q1 + q2 + vh) - 2*c2z*dq2*dvh*m2*np.sin(q1 + q2 + vh) - 2*c4z*dq3*dvh*m4*np.sin(q3 + q4 + vh) - 2*c4z*dq4*dvh*m4*np.sin(q3 + q4 + vh) - 2*L1x*dq1*dvh*m2*np.cos(q1 + vh) - 2*L3x*dq3*dvh*m4*np.cos(q3 + vh) - 2*L1z*dq1*dvh*m2*np.sin(q1 + vh) - 2*L3z*dq3*dvh*m4*np.sin(q3 + vh) - 2*c1x*dq1*dvh*m1*np.cos(q1 + vh) - 2*c3x*dq3*dvh*m3*np.cos(q3 + vh) - 2*c1z*dq1*dvh*m1*np.sin(q1 + vh) - 2*c3z*dq3*dvh*m3*np.sin(q3 + vh)
	CG[1,0] = g*m0 + g*m1 + g*m2 + g*m3 + g*m4 - c2z*dq1**2*m2*np.cos(q1 + q2 + vh) - c2z*dq2**2*m2*np.cos(q1 + q2 + vh) - c4z*dq3**2*m4*np.cos(q3 + q4 + vh) - c4z*dq4**2*m4*np.cos(q3 + q4 + vh) - c2z*dvh**2*m2*np.cos(q1 + q2 + vh) - c4z*dvh**2*m4*np.cos(q3 + q4 + vh) + c2x*dq1**2*m2*np.sin(q1 + q2 + vh) + c2x*dq2**2*m2*np.sin(q1 + q2 + vh) + c4x*dq3**2*m4*np.sin(q3 + q4 + vh) + c4x*dq4**2*m4*np.sin(q3 + q4 + vh) + c2x*dvh**2*m2*np.sin(q1 + q2 + vh) + c4x*dvh**2*m4*np.sin(q3 + q4 + vh) - L1z*dq1**2*m2*np.cos(q1 + vh) - L3z*dq3**2*m4*np.cos(q3 + vh) - L1z*dvh**2*m2*np.cos(q1 + vh) - L3z*dvh**2*m4*np.cos(q3 + vh) + L1x*dq1**2*m2*np.sin(q1 + vh) + L3x*dq3**2*m4*np.sin(q3 + vh) + L1x*dvh**2*m2*np.sin(q1 + vh) + L3x*dvh**2*m4*np.sin(q3 + vh) - c1z*dq1**2*m1*np.cos(q1 + vh) - c3z*dq3**2*m3*np.cos(q3 + vh) - c1z*dvh**2*m1*np.cos(q1 + vh) - c3z*dvh**2*m3*np.cos(q3 + vh) + c1x*dq1**2*m1*np.sin(q1 + vh) + c3x*dq3**2*m3*np.sin(q3 + vh) + c1x*dvh**2*m1*np.sin(q1 + vh) + c3x*dvh**2*m3*np.sin(q3 + vh) - L0z*dvh**2*m1*np.cos(vh) - L0z*dvh**2*m2*np.cos(vh) - L0z*dvh**2*m3*np.cos(vh) - L0z*dvh**2*m4*np.cos(vh) + L0x*dvh**2*m1*np.sin(vh) + L0x*dvh**2*m2*np.sin(vh) - L0x*dvh**2*m3*np.sin(vh) - L0x*dvh**2*m4*np.sin(vh) - c0z*dvh**2*m0*np.cos(vh) + c0x*dvh**2*m0*np.sin(vh) - 2*c2z*dq1*dq2*m2*np.cos(q1 + q2 + vh) - 2*c4z*dq3*dq4*m4*np.cos(q3 + q4 + vh) - 2*c2z*dq1*dvh*m2*np.cos(q1 + q2 + vh) - 2*c2z*dq2*dvh*m2*np.cos(q1 + q2 + vh) - 2*c4z*dq3*dvh*m4*np.cos(q3 + q4 + vh) - 2*c4z*dq4*dvh*m4*np.cos(q3 + q4 + vh) + 2*c2x*dq1*dq2*m2*np.sin(q1 + q2 + vh) + 2*c4x*dq3*dq4*m4*np.sin(q3 + q4 + vh) + 2*c2x*dq1*dvh*m2*np.sin(q1 + q2 + vh) + 2*c2x*dq2*dvh*m2*np.sin(q1 + q2 + vh) + 2*c4x*dq3*dvh*m4*np.sin(q3 + q4 + vh) + 2*c4x*dq4*dvh*m4*np.sin(q3 + q4 + vh) - 2*L1z*dq1*dvh*m2*np.cos(q1 + vh) - 2*L3z*dq3*dvh*m4*np.cos(q3 + vh) + 2*L1x*dq1*dvh*m2*np.sin(q1 + vh) + 2*L3x*dq3*dvh*m4*np.sin(q3 + vh) - 2*c1z*dq1*dvh*m1*np.cos(q1 + vh) - 2*c3z*dq3*dvh*m3*np.cos(q3 + vh) + 2*c1x*dq1*dvh*m1*np.sin(q1 + vh) + 2*c3x*dq3*dvh*m3*np.sin(q3 + vh)
	CG[2,0] = L0x*g*m3*np.cos(vh) - L3x*g*m4*np.cos(q3 + vh) - L1z*g*m2*np.sin(q1 + vh) - L3z*g*m4*np.sin(q3 + vh) - c1x*g*m1*np.cos(q1 + vh) - c3x*g*m3*np.cos(q3 + vh) - c1z*g*m1*np.sin(q1 + vh) - c3z*g*m3*np.sin(q3 + vh) - L0x*g*m1*np.cos(vh) - L0x*g*m2*np.cos(vh) - L1x*g*m2*np.cos(q1 + vh) + L0x*g*m4*np.cos(vh) - L0z*g*m1*np.sin(vh) - L0z*g*m2*np.sin(vh) - L0z*g*m3*np.sin(vh) - L0z*g*m4*np.sin(vh) - c0x*g*m0*np.cos(vh) - c0z*g*m0*np.sin(vh) - c2x*g*m2*np.cos(q1 + q2 + vh) - c4x*g*m4*np.cos(q3 + q4 + vh) - c2z*g*m2*np.sin(q1 + q2 + vh) - c4z*g*m4*np.sin(q3 + q4 + vh) - L0x*L1x*dq1**2*m2*np.sin(q1) + L0x*L3x*dq3**2*m4*np.sin(q3) - L0z*L1z*dq1**2*m2*np.sin(q1) - L0z*L3z*dq3**2*m4*np.sin(q3) + L0x*c1z*dq1**2*m1*np.cos(q1) - L0z*c1x*dq1**2*m1*np.cos(q1) + L1x*c2z*dq2**2*m2*np.cos(q2) - L1z*c2x*dq2**2*m2*np.cos(q2) - L0x*c3z*dq3**2*m3*np.cos(q3) - L0z*c3x*dq3**2*m3*np.cos(q3) + L3x*c4z*dq4**2*m4*np.cos(q4) - L3z*c4x*dq4**2*m4*np.cos(q4) - L0x*c1x*dq1**2*m1*np.sin(q1) - L1x*c2x*dq2**2*m2*np.sin(q2) + L0x*c3x*dq3**2*m3*np.sin(q3) - L3x*c4x*dq4**2*m4*np.sin(q4) - L0z*c1z*dq1**2*m1*np.sin(q1) - L1z*c2z*dq2**2*m2*np.sin(q2) - L0z*c3z*dq3**2*m3*np.sin(q3) - L3z*c4z*dq4**2*m4*np.sin(q4) + L0x*c2z*dq1**2*m2*np.cos(q1 + q2) - L0z*c2x*dq1**2*m2*np.cos(q1 + q2) + L0x*c2z*dq2**2*m2*np.cos(q1 + q2) - L0z*c2x*dq2**2*m2*np.cos(q1 + q2) - L0x*c4z*dq3**2*m4*np.cos(q3 + q4) - L0z*c4x*dq3**2*m4*np.cos(q3 + q4) - L0x*c4z*dq4**2*m4*np.cos(q3 + q4) - L0z*c4x*dq4**2*m4*np.cos(q3 + q4) - L0x*c2x*dq1**2*m2*np.sin(q1 + q2) - L0x*c2x*dq2**2*m2*np.sin(q1 + q2) + L0x*c4x*dq3**2*m4*np.sin(q3 + q4) + L0x*c4x*dq4**2*m4*np.sin(q3 + q4) - L0z*c2z*dq1**2*m2*np.sin(q1 + q2) - L0z*c2z*dq2**2*m2*np.sin(q1 + q2) - L0z*c4z*dq3**2*m4*np.sin(q3 + q4) - L0z*c4z*dq4**2*m4*np.sin(q3 + q4) + L0x*L1z*dq1**2*m2*np.cos(q1) - L1x*L0z*dq1**2*m2*np.cos(q1) - L0x*L3z*dq3**2*m4*np.cos(q3) - L3x*L0z*dq3**2*m4*np.cos(q3) + 2*L0x*L1z*dq1*dvh*m2*np.cos(q1) - 2*L1x*L0z*dq1*dvh*m2*np.cos(q1) - 2*L0x*L3z*dq3*dvh*m4*np.cos(q3) - 2*L3x*L0z*dq3*dvh*m4*np.cos(q3) - 2*L0x*L1x*dq1*dvh*m2*np.sin(q1) + 2*L0x*L3x*dq3*dvh*m4*np.sin(q3) - 2*L0z*L1z*dq1*dvh*m2*np.sin(q1) - 2*L0z*L3z*dq3*dvh*m4*np.sin(q3) + 2*L1x*c2z*dq1*dq2*m2*np.cos(q2) - 2*L1z*c2x*dq1*dq2*m2*np.cos(q2) + 2*L3x*c4z*dq3*dq4*m4*np.cos(q4) - 2*L3z*c4x*dq3*dq4*m4*np.cos(q4) + 2*L0x*c1z*dq1*dvh*m1*np.cos(q1) - 2*L0z*c1x*dq1*dvh*m1*np.cos(q1) + 2*L1x*c2z*dq2*dvh*m2*np.cos(q2) - 2*L1z*c2x*dq2*dvh*m2*np.cos(q2) - 2*L0x*c3z*dq3*dvh*m3*np.cos(q3) - 2*L0z*c3x*dq3*dvh*m3*np.cos(q3) + 2*L3x*c4z*dq4*dvh*m4*np.cos(q4) - 2*L3z*c4x*dq4*dvh*m4*np.cos(q4) - 2*L1x*c2x*dq1*dq2*m2*np.sin(q2) - 2*L3x*c4x*dq3*dq4*m4*np.sin(q4) - 2*L1z*c2z*dq1*dq2*m2*np.sin(q2) - 2*L3z*c4z*dq3*dq4*m4*np.sin(q4) - 2*L0x*c1x*dq1*dvh*m1*np.sin(q1) - 2*L1x*c2x*dq2*dvh*m2*np.sin(q2) + 2*L0x*c3x*dq3*dvh*m3*np.sin(q3) - 2*L3x*c4x*dq4*dvh*m4*np.sin(q4) - 2*L0z*c1z*dq1*dvh*m1*np.sin(q1) - 2*L1z*c2z*dq2*dvh*m2*np.sin(q2) - 2*L0z*c3z*dq3*dvh*m3*np.sin(q3) - 2*L3z*c4z*dq4*dvh*m4*np.sin(q4) + 2*L0x*c2z*dq1*dq2*m2*np.cos(q1 + q2) - 2*L0z*c2x*dq1*dq2*m2*np.cos(q1 + q2) - 2*L0x*c4z*dq3*dq4*m4*np.cos(q3 + q4) - 2*L0z*c4x*dq3*dq4*m4*np.cos(q3 + q4) + 2*L0x*c2z*dq1*dvh*m2*np.cos(q1 + q2) - 2*L0z*c2x*dq1*dvh*m2*np.cos(q1 + q2) + 2*L0x*c2z*dq2*dvh*m2*np.cos(q1 + q2) - 2*L0z*c2x*dq2*dvh*m2*np.cos(q1 + q2) - 2*L0x*c4z*dq3*dvh*m4*np.cos(q3 + q4) - 2*L0z*c4x*dq3*dvh*m4*np.cos(q3 + q4) - 2*L0x*c4z*dq4*dvh*m4*np.cos(q3 + q4) - 2*L0z*c4x*dq4*dvh*m4*np.cos(q3 + q4) - 2*L0x*c2x*dq1*dq2*m2*np.sin(q1 + q2) + 2*L0x*c4x*dq3*dq4*m4*np.sin(q3 + q4) - 2*L0z*c2z*dq1*dq2*m2*np.sin(q1 + q2) - 2*L0z*c4z*dq3*dq4*m4*np.sin(q3 + q4) - 2*L0x*c2x*dq1*dvh*m2*np.sin(q1 + q2) - 2*L0x*c2x*dq2*dvh*m2*np.sin(q1 + q2) + 2*L0x*c4x*dq3*dvh*m4*np.sin(q3 + q4) + 2*L0x*c4x*dq4*dvh*m4*np.sin(q3 + q4) - 2*L0z*c2z*dq1*dvh*m2*np.sin(q1 + q2) - 2*L0z*c2z*dq2*dvh*m2*np.sin(q1 + q2) - 2*L0z*c4z*dq3*dvh*m4*np.sin(q3 + q4) - 2*L0z*c4z*dq4*dvh*m4*np.sin(q3 + q4)
	CG[3,0] = L0x*L1x*dvh**2*m2*np.sin(q1) - L1z*g*m2*np.sin(q1 + vh) - c1x*g*m1*np.cos(q1 + vh) - c1z*g*m1*np.sin(q1 + vh) - c2x*g*m2*np.cos(q1 + q2 + vh) - c2z*g*m2*np.sin(q1 + q2 + vh) - L1x*g*m2*np.cos(q1 + vh) + L0z*L1z*dvh**2*m2*np.sin(q1) + L1x*c2z*dq2**2*m2*np.cos(q2) - L1z*c2x*dq2**2*m2*np.cos(q2) - L0x*c1z*dvh**2*m1*np.cos(q1) + L0z*c1x*dvh**2*m1*np.cos(q1) - L1x*c2x*dq2**2*m2*np.sin(q2) - L1z*c2z*dq2**2*m2*np.sin(q2) + L0x*c1x*dvh**2*m1*np.sin(q1) + L0z*c1z*dvh**2*m1*np.sin(q1) - L0x*c2z*dvh**2*m2*np.cos(q1 + q2) + L0z*c2x*dvh**2*m2*np.cos(q1 + q2) + L0x*c2x*dvh**2*m2*np.sin(q1 + q2) + L0z*c2z*dvh**2*m2*np.sin(q1 + q2) - L0x*L1z*dvh**2*m2*np.cos(q1) + L1x*L0z*dvh**2*m2*np.cos(q1) + 2*L1x*c2z*dq1*dq2*m2*np.cos(q2) - 2*L1z*c2x*dq1*dq2*m2*np.cos(q2) + 2*L1x*c2z*dq2*dvh*m2*np.cos(q2) - 2*L1z*c2x*dq2*dvh*m2*np.cos(q2) - 2*L1x*c2x*dq1*dq2*m2*np.sin(q2) - 2*L1z*c2z*dq1*dq2*m2*np.sin(q2) - 2*L1x*c2x*dq2*dvh*m2*np.sin(q2) - 2*L1z*c2z*dq2*dvh*m2*np.sin(q2)
	CG[4,0] = m2*(L0z*c2x*dvh**2*np.cos(q1 + q2) - c2z*g*np.sin(q1 + q2 + vh) - L0x*c2z*dvh**2*np.cos(q1 + q2) - c2x*g*np.cos(q1 + q2 + vh) + L0x*c2x*dvh**2*np.sin(q1 + q2) + L0z*c2z*dvh**2*np.sin(q1 + q2) - L1x*c2z*dq1**2*np.cos(q2) + L1z*c2x*dq1**2*np.cos(q2) - L1x*c2z*dvh**2*np.cos(q2) + L1z*c2x*dvh**2*np.cos(q2) + L1x*c2x*dq1**2*np.sin(q2) + L1z*c2z*dq1**2*np.sin(q2) + L1x*c2x*dvh**2*np.sin(q2) + L1z*c2z*dvh**2*np.sin(q2) - 2*L1x*c2z*dq1*dvh*np.cos(q2) + 2*L1z*c2x*dq1*dvh*np.cos(q2) + 2*L1x*c2x*dq1*dvh*np.sin(q2) + 2*L1z*c2z*dq1*dvh*np.sin(q2))
	CG[5,0] = L0z*L3z*dvh**2*m4*np.sin(q3) - L3z*g*m4*np.sin(q3 + vh) - c3x*g*m3*np.cos(q3 + vh) - c3z*g*m3*np.sin(q3 + vh) - c4x*g*m4*np.cos(q3 + q4 + vh) - c4z*g*m4*np.sin(q3 + q4 + vh) - L0x*L3x*dvh**2*m4*np.sin(q3) - L3x*g*m4*np.cos(q3 + vh) + L3x*c4z*dq4**2*m4*np.cos(q4) - L3z*c4x*dq4**2*m4*np.cos(q4) + L0x*c3z*dvh**2*m3*np.cos(q3) + L0z*c3x*dvh**2*m3*np.cos(q3) - L3x*c4x*dq4**2*m4*np.sin(q4) - L3z*c4z*dq4**2*m4*np.sin(q4) - L0x*c3x*dvh**2*m3*np.sin(q3) + L0z*c3z*dvh**2*m3*np.sin(q3) + L0x*c4z*dvh**2*m4*np.cos(q3 + q4) + L0z*c4x*dvh**2*m4*np.cos(q3 + q4) - L0x*c4x*dvh**2*m4*np.sin(q3 + q4) + L0z*c4z*dvh**2*m4*np.sin(q3 + q4) + L0x*L3z*dvh**2*m4*np.cos(q3) + L3x*L0z*dvh**2*m4*np.cos(q3) + 2*L3x*c4z*dq3*dq4*m4*np.cos(q4) - 2*L3z*c4x*dq3*dq4*m4*np.cos(q4) + 2*L3x*c4z*dq4*dvh*m4*np.cos(q4) - 2*L3z*c4x*dq4*dvh*m4*np.cos(q4) - 2*L3x*c4x*dq3*dq4*m4*np.sin(q4) - 2*L3z*c4z*dq3*dq4*m4*np.sin(q4) - 2*L3x*c4x*dq4*dvh*m4*np.sin(q4) - 2*L3z*c4z*dq4*dvh*m4*np.sin(q4)
	CG[6,0] = m4*(L0x*c4z*dvh**2*np.cos(q3 + q4) - c4z*g*np.sin(q3 + q4 + vh) - c4x*g*np.cos(q3 + q4 + vh) + L0z*c4x*dvh**2*np.cos(q3 + q4) - L0x*c4x*dvh**2*np.sin(q3 + q4) + L0z*c4z*dvh**2*np.sin(q3 + q4) - L3x*c4z*dq3**2*np.cos(q4) + L3z*c4x*dq3**2*np.cos(q4) - L3x*c4z*dvh**2*np.cos(q4) + L3z*c4x*dvh**2*np.cos(q4) + L3x*c4x*dq3**2*np.sin(q4) + L3z*c4z*dq3**2*np.sin(q4) + L3x*c4x*dvh**2*np.sin(q4) + L3z*c4z*dvh**2*np.sin(q4) - 2*L3x*c4z*dq3*dvh*np.cos(q4) + 2*L3z*c4x*dq3*dvh*np.cos(q4) + 2*L3x*c4x*dq3*dvh*np.sin(q4) + 2*L3z*c4z*dq3*dvh*np.sin(q4))
	# Task space jacobian
	J = np.zeros((3,7))
	J[0,0] = 1.0
	J[1,1] = 1.0
	J[2,2] = 1.0
	# Task space jacobian derivative times velocity
	dJdq = np.zeros((3,1))
	# Support space jacobian
	Js = np.zeros((4,7))
	Js[0,0] = 1.0
	Js[0,2] = L1z*np.cos(q1 + vh) - L1x*np.sin(q1 + vh) + L0z*np.cos(vh) - L0x*np.sin(vh) + L2z*np.cos(q1 + q2 + vh) - L2x*np.sin(q1 + q2 + vh)
	Js[0,3] = L1z*np.cos(q1 + vh) - L1x*np.sin(q1 + vh) + L2z*np.cos(q1 + q2 + vh) - L2x*np.sin(q1 + q2 + vh)
	Js[0,4] = L2z*np.cos(q1 + q2 + vh) - L2x*np.sin(q1 + q2 + vh)
	Js[1,1] = 1.0
	Js[1,2] = - L1x*np.cos(q1 + vh) - L1z*np.sin(q1 + vh) - L0x*np.cos(vh) - L0z*np.sin(vh) - L2x*np.cos(q1 + q2 + vh) - L2z*np.sin(q1 + q2 + vh)
	Js[1,3] = - L1x*np.cos(q1 + vh) - L1z*np.sin(q1 + vh) - L2x*np.cos(q1 + q2 + vh) - L2z*np.sin(q1 + q2 + vh)
	Js[1,4] = - L2x*np.cos(q1 + q2 + vh) - L2z*np.sin(q1 + q2 + vh)
	Js[2,0] = 1.0
	Js[2,2] = L3z*np.cos(q3 + vh) - L3x*np.sin(q3 + vh) + L0z*np.cos(vh) + L0x*np.sin(vh) + L4z*np.cos(q3 + q4 + vh) - L4x*np.sin(q3 + q4 + vh)
	Js[2,5] = L3z*np.cos(q3 + vh) - L3x*np.sin(q3 + vh) + L4z*np.cos(q3 + q4 + vh) - L4x*np.sin(q3 + q4 + vh)
	Js[2,6] = L4z*np.cos(q3 + q4 + vh) - L4x*np.sin(q3 + q4 + vh)
	Js[3,1] = 1.0
	Js[3,2] = L0x*np.cos(vh) - L3z*np.sin(q3 + vh) - L3x*np.cos(q3 + vh) - L0z*np.sin(vh) - L4x*np.cos(q3 + q4 + vh) - L4z*np.sin(q3 + q4 + vh)
	Js[3,5] = - L3x*np.cos(q3 + vh) - L3z*np.sin(q3 + vh) - L4x*np.cos(q3 + q4 + vh) - L4z*np.sin(q3 + q4 + vh)
	Js[3,6] = - L4x*np.cos(q3 + q4 + vh) - L4z*np.sin(q3 + q4 + vh)
	# support space jacobian derivative times velocity
	dJsdq = np.zeros((4,1))
	dJsdq[0,0] = - L1x*dq1**2*np.cos(q1 + vh) - L1x*dvh**2*np.cos(q1 + vh) - L1z*dq1**2*np.sin(q1 + vh) - L1z*dvh**2*np.sin(q1 + vh) - L0x*dvh**2*np.cos(vh) - L0z*dvh**2*np.sin(vh) - L2x*dq1**2*np.cos(q1 + q2 + vh) - L2x*dq2**2*np.cos(q1 + q2 + vh) - L2x*dvh**2*np.cos(q1 + q2 + vh) - L2z*dq1**2*np.sin(q1 + q2 + vh) - L2z*dq2**2*np.sin(q1 + q2 + vh) - L2z*dvh**2*np.sin(q1 + q2 + vh) - 2*L1x*dq1*dvh*np.cos(q1 + vh) - 2*L1z*dq1*dvh*np.sin(q1 + vh) - 2*L2x*dq1*dq2*np.cos(q1 + q2 + vh) - 2*L2x*dq1*dvh*np.cos(q1 + q2 + vh) - 2*L2x*dq2*dvh*np.cos(q1 + q2 + vh) - 2*L2z*dq1*dq2*np.sin(q1 + q2 + vh) - 2*L2z*dq1*dvh*np.sin(q1 + q2 + vh) - 2*L2z*dq2*dvh*np.sin(q1 + q2 + vh)
	dJsdq[1,0] = L1x*dq1**2*np.sin(q1 + vh) - L1z*dvh**2*np.cos(q1 + vh) - L1z*dq1**2*np.cos(q1 + vh) + L1x*dvh**2*np.sin(q1 + vh) - L0z*dvh**2*np.cos(vh) + L0x*dvh**2*np.sin(vh) - L2z*dq1**2*np.cos(q1 + q2 + vh) - L2z*dq2**2*np.cos(q1 + q2 + vh) - L2z*dvh**2*np.cos(q1 + q2 + vh) + L2x*dq1**2*np.sin(q1 + q2 + vh) + L2x*dq2**2*np.sin(q1 + q2 + vh) + L2x*dvh**2*np.sin(q1 + q2 + vh) - 2*L1z*dq1*dvh*np.cos(q1 + vh) + 2*L1x*dq1*dvh*np.sin(q1 + vh) - 2*L2z*dq1*dq2*np.cos(q1 + q2 + vh) - 2*L2z*dq1*dvh*np.cos(q1 + q2 + vh) - 2*L2z*dq2*dvh*np.cos(q1 + q2 + vh) + 2*L2x*dq1*dq2*np.sin(q1 + q2 + vh) + 2*L2x*dq1*dvh*np.sin(q1 + q2 + vh) + 2*L2x*dq2*dvh*np.sin(q1 + q2 + vh)
	dJsdq[2,0] = L0x*dvh**2*np.cos(vh) - L3x*dvh**2*np.cos(q3 + vh) - L3z*dq3**2*np.sin(q3 + vh) - L3z*dvh**2*np.sin(q3 + vh) - L3x*dq3**2*np.cos(q3 + vh) - L0z*dvh**2*np.sin(vh) - L4x*dq3**2*np.cos(q3 + q4 + vh) - L4x*dq4**2*np.cos(q3 + q4 + vh) - L4x*dvh**2*np.cos(q3 + q4 + vh) - L4z*dq3**2*np.sin(q3 + q4 + vh) - L4z*dq4**2*np.sin(q3 + q4 + vh) - L4z*dvh**2*np.sin(q3 + q4 + vh) - 2*L3x*dq3*dvh*np.cos(q3 + vh) - 2*L3z*dq3*dvh*np.sin(q3 + vh) - 2*L4x*dq3*dq4*np.cos(q3 + q4 + vh) - 2*L4x*dq3*dvh*np.cos(q3 + q4 + vh) - 2*L4x*dq4*dvh*np.cos(q3 + q4 + vh) - 2*L4z*dq3*dq4*np.sin(q3 + q4 + vh) - 2*L4z*dq3*dvh*np.sin(q3 + q4 + vh) - 2*L4z*dq4*dvh*np.sin(q3 + q4 + vh)
	dJsdq[3,0] = L3x*dq3**2*np.sin(q3 + vh) - L3z*dvh**2*np.cos(q3 + vh) - L3z*dq3**2*np.cos(q3 + vh) + L3x*dvh**2*np.sin(q3 + vh) - L0z*dvh**2*np.cos(vh) - L0x*dvh**2*np.sin(vh) - L4z*dq3**2*np.cos(q3 + q4 + vh) - L4z*dq4**2*np.cos(q3 + q4 + vh) - L4z*dvh**2*np.cos(q3 + q4 + vh) + L4x*dq3**2*np.sin(q3 + q4 + vh) + L4x*dq4**2*np.sin(q3 + q4 + vh) + L4x*dvh**2*np.sin(q3 + q4 + vh) - 2*L3z*dq3*dvh*np.cos(q3 + vh) + 2*L3x*dq3*dvh*np.sin(q3 + vh) - 2*L4z*dq3*dq4*np.cos(q3 + q4 + vh) - 2*L4z*dq3*dvh*np.cos(q3 + q4 + vh) - 2*L4z*dq4*dvh*np.cos(q3 + q4 + vh) + 2*L4x*dq3*dq4*np.sin(q3 + q4 + vh) + 2*L4x*dq3*dvh*np.sin(q3 + q4 + vh) + 2*L4x*dq4*dvh*np.sin(q3 + q4 + vh)
	# Actuator selection matrix
	Sa = np.zeros((4,7))
	Sa[0,3] = 1.0
	Sa[1,4] = 1.0
	Sa[2,5] = 1.0
	Sa[3,6] = 1.0
	# Desired task coordinate accelerations
	ddX = np.zeros((3,1))
	ddX[0,0] = ddx
	ddX[1,0] = ddz
	ddX[2,0] = ddvh

	#### Do the inverse dynamics part!
	Hinv = np.linalg.inv(H)
	lambdas = np.linalg.inv(np.dot(Js, np.dot(Hinv, Js.T)))
	JsbarT = np.dot(lambdas, np.dot(Js,Hinv))
	Ns = np.eye(7) - np.dot(Js.T, JsbarT)
	lambdat = np.linalg.inv(np.dot(J, np.dot(Hinv, np.dot(Ns, J.T))))
	JbarT = np.dot(lambdat, np.dot(J, np.dot(Hinv, Ns)))
	murho = np.dot(JbarT, CG) - np.dot(lambdat, dJdq) + np.dot(lambdat, np.dot(J, np.dot(Hinv, np.dot(Js.T, np.dot(lambdas, dJsdq)))))
	F = np.dot(lambdat, ddX) + murho
	# tau = np.dot(np.linalg.inv(np.dot(JbarT, Sa.T)), F)
	tau = np.dot(np.linalg.pinv(np.dot(JbarT, Sa.T)), F)
	return tau

if __name__ == "__main__":
	main()