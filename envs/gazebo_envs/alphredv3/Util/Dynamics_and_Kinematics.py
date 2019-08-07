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

from numba import njit
import numpy as np
from ..Settings.Macros_ALPHREDV3 import *
from ..Util import MathFcn as MF
# import Util.MathFcn as MF
import pdb

"""
Dynamics equations and Kinematics equations for the ALPHRED V3 platform. Uses Numba to optimize performance.
"""


# @njit()
def forward_kinematics(q):
    clocking = [-np.pi/4.0, np.pi/4.0, 3.0*np.pi/4.0, -3.0*np.pi/4.0]
    limbs = np.zeros((3, 4))

    for i in range(0, 4):
        q0 = clocking[i]
        q1 = q[i * 3][0]
        q2 = q[1 + i * 3][0]
        q3 = q[2 + i * 3][0]

        limbs[:, i] = np.array([ROTOR_OFFSET * np.cos(q0 + q1) + BODY_OFFSET * np.cos(q0) +
                                LENGTH_FEMUR * np.cos(q0 + q1) * np.cos(q2) + LENGTH_TIBIA * np.cos(q0 + q1) * np.cos(q2 + q3),
                                ROTOR_OFFSET * np.sin(q0 + q1) + BODY_OFFSET * np.sin(q0) + 
                                LENGTH_FEMUR * np.sin(q0 + q1) * np.cos(q2) + LENGTH_TIBIA * np.sin(q0 + q1) * np.cos(q2 + q3) ,
                                - LENGTH_TIBIA * np.sin(q2 + q3) - LENGTH_FEMUR * np.sin(q2)])

    return limbs


# @njit()
def inverse_kinematics(vec,limb_num):
    clocking = [-np.pi/4.0, np.pi/4.0, 3.0*np.pi/4.0, -3.0*np.pi/4.0]

    s = np.sin(clocking[limb_num-1])
    c = np.cos(clocking[limb_num-1])
    rot = np.zeros((3,3))
    rot[0,0] = c
    rot[0,1] = s
    rot[1,0] = -s
    rot[1,1] = c
    rot[2,2] = 1.0
    r = np.dot(rot, vec)
    hip_yaw = np.arctan2(r[1,0], r[0,0] - BODY_OFFSET)
    knee_pitch = 0.0
    hip_pitch = 0.0

    X = r[0,0] - BODY_OFFSET - ROTOR_OFFSET * np.cos(hip_yaw)
    Y = r[1,0] - ROTOR_OFFSET * np.sin(hip_yaw)
    Z = r[2,0]
    L = (X ** 2.0 + Y ** 2.0) ** 0.5
    D = (L ** 2.0 + Z ** 2.0) ** 0.5
    # if D > (LENGTH_FEMUR + LENGTH_TIBIA):
    #    raise IKError("D ({}) is greater than the sum of the length of the two links({}).".format(D, LENGTH_FEMUR+LENGTH_TIBIA))
    if X < 0:
        if Z < 0:
            sigma = -np.pi / 2.0 - (np.pi / 2.0 + np.arctan2(Z, L))
        else:
            sigma = np.pi / 2.0 + (np.pi / 2.0 - np.arctan2(Z, L))
    else:
        sigma = np.arctan2(Z, L)
    if D > (LENGTH_FEMUR + LENGTH_TIBIA):
        knee_pitch = 0.0
        hip_pitch = np.arccos(old_div(L,D))
    else:
        theta = np.arccos(old_div((LENGTH_FEMUR ** 2.0 + D ** 2.0 - LENGTH_TIBIA ** 2.0), (2 * LENGTH_FEMUR * D)))
        hip_pitch = -(sigma + theta)
        knee_pitch = np.pi - np.arccos(
            old_div((LENGTH_FEMUR ** 2.0 + LENGTH_TIBIA ** 2.0 - D ** 2.0), (2 * LENGTH_FEMUR * LENGTH_TIBIA)))
    return hip_yaw, hip_pitch, knee_pitch

# @njit()
def jacobian_all_limb(q):
    """
    Calculates the jacobians for all four limbs
    :param q:   (N_DOF_LIMB*N_LIMB x 1) Concatenated vector of all joint positions (12x1)
    :return J:  (N_LIMB x N_DOF_LIMB x N_DOF_LIMB) 3d array of jacobians for each limb (4x3x3)
    """
    clocking = [-np.pi/4.0, np.pi/4.0, 3.0*np.pi/4.0, -3.0*np.pi/4.0]
    J = np.zeros((4,3,3))
    for i_limb in range(4):
        q0 = clocking[i_limb]
        q1 = q[i_limb * 3][0]
        q2 = q[1 + i_limb * 3][0]
        q3 = q[2 + i_limb * 3][0]
        J[i_limb,0,0] = -np.sin(q0 + q1)*(ROTOR_OFFSET + LENGTH_FEMUR*np.cos(q2) + LENGTH_TIBIA*np.cos(q2 + q3))
        J[i_limb,0,1] = -np.cos(q0 + q1)*(LENGTH_FEMUR*np.sin(q2) + LENGTH_TIBIA*np.sin(q2 + q3))
        J[i_limb,0,2] = -np.cos(q0 + q1)*LENGTH_TIBIA*np.sin(q2 + q3)
        J[i_limb,1,0] =  np.cos(q0 + q1)*(ROTOR_OFFSET + LENGTH_FEMUR*np.cos(q2) + LENGTH_TIBIA*np.cos(q2 + q3))
        J[i_limb,1,1] = -np.sin(q0 + q1)*(LENGTH_FEMUR*np.sin(q2) + LENGTH_TIBIA*np.sin(q2 + q3))
        J[i_limb,1,2] = -np.sin(q0 + q1)*LENGTH_TIBIA*np.sin(q2 + q3)
        J[i_limb,2,0] = 0
        J[i_limb,2,1] = -LENGTH_FEMUR*np.cos(q2) - LENGTH_TIBIA*np.cos(q2 + q3)
        J[i_limb,2,2] = -LENGTH_TIBIA*np.cos(q2 + q3)
    return J

# @njit()
def jacobian_velocity(q, dq):
    """
    Calculates the end effector (foot) velocity using the current joint state
    :param q:   (N_DOF_LIMB*N_LIMB x 1) Concatenated vector of all joint positions (12x1)
    :param dq:  (N_DOF_LIMB*N_LIMB x 1) Concatenated vector of all joint velocities (12x1)
    :return dx: (3*N_LIMB x 1) Concatenated vector of all foot velocities (12x1)
    """
    J = jacobian_all_limb(q)
    dx = np.zeros((12, 1))
    # pdb.set_trace()
    for i_limb in range(4):
        dx[i_limb*3:(i_limb+1)*3,0] = J[i_limb].dot(dq[i_limb*3:(i_limb+1)*3,0])
    return dx

# @njit()
def jacobian_torque(q, F):
    """
    Calculates the joint torques given foot forces and joint position
    :param q:   (N_DOF_LIMB*N_LIMB x 1) Concatenated vector of all joint positions (12x1)
    :param F:   (3*N_LIMB x 1) Concatenated vector of end effector forces (12x1)
    :return tau:(N_DOF_LIMB*N_LIMB x 1) Concatenated vector of all joint torques (12,1)
    """
    J = jacobian_all_limb(q)
    tau = np.zeros((12, 1))
    for i_limb in range(4):
        tau[i_limb*3:(i_limb+1)*3,0] = J[i_limb].T.dot(F[i_limb*3:(i_limb+1)*3,0])
    return tau
