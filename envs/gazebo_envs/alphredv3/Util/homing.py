#!usr/bin/env python
from __future__ import print_function
from builtins import str
from builtins import input
from builtins import range
__author__ = "Joshua Hooks"
__email__ = "hooksjrose@gmail.com"
__copyright__ = "Copyright 2019 RoMeLa"
__date__ = "June 12, 2019"
__version__ = "0.0.1"
__status__ = "Prototype"

"""
This script is used to home the joints on ALPHRED V3.

    - Standard use is to enter in the desired joint to be calibrated while using the calibration tools.
    - Users can also hand input offsets.
    - all offsets are logged in homing_logs/joint_#.txt
"""

import time
import datetime
import numpy as np
from pybear import Manager
from Settings.Macros_ALPHREDV3 import *
import Settings.Robot as ALPHRED

pdm1 = Manager.BEAR(port=ALPHRED.pdm_port1, baudrate=ALPHRED.baudrate)

def main():
    stop = False

    while not stop:
        print("========================")
        print("      Homing Tool       ")
        print("========================")
        print("Commands:")
        print("(#[joint number])\t\tCalibrate the given joint")
        print("(#[joint number] #[offset])\t\tSet the given joint to the given offset value")
        print("(p)\t\tPrint out current position of all joints")
        print("(w)\t\tWrite out current offsets for all joints")
        print("(q)\t\tQuit")
        print()
        cmd = str(input("Input Command: "))
        if 0 < len(cmd) < 3:
            if cmd == '1' or cmd == '4' or cmd == '7' or cmd == '10':
                joint = int(cmd)
                home_hip_yaw_joint(joint)
            elif cmd == '2' or cmd == '5' or cmd == '8' or cmd == '11':
                joint = int(cmd)
                home_hip_pitch_joint(joint)
            elif cmd == '3' or cmd == '6' or cmd == '9' or cmd == '12':
                joint = int(cmd)
                home_knee_pitch_joint(joint-1, joint)
            elif cmd == 'p':
                read_positions()
            elif cmd == 'w':
                write_offsets()
            elif cmd == 'q':
                stop = True
            else:
                print ("Invalid input!")
                print()
                print()
        elif len(cmd) >= 3:
            cmd_list = cmd.split(" ")
            if len(cmd_list) == 2:
                try:
                    joint_num = int(cmd_list[0])
                    offset = int(cmd_list[1])
                    pdm1.set_homing_offset((joint_num, offset))
                    pdm1.save_config(joint_num)

                    # write to log file
                    filename = "homing_logs/joint_" + cmd_list[0] + ".txt"
                    log = open(filename, "a+")
                    now = datetime.datetime.now()
                    data_entry = now.strftime("%Y-%m-%d %H:%M") + ", user entry, " + cmd_list[1] + "\n"
                    log.write(data_entry)
                    log.close()

                    print()
                    print ("Joint %s offset was successfully set to %s" %(joint_num, offset))
                    print()
                    print()
                except ValueError:
                    print("Invalid input, expecting an integer for the joint number and an integer for the offset.")
                    print()
                    print()

            elif len(cmd_list) == 1:
                print ("Invalid joint number, must be between 1-12!")
                print()
                print()
            else:
                print ("To many inputs!")
                print()
                print()


def write_offsets():
    """
    Prints out the current offsets for the user to see and also writes them to the log files.
    """

    offsets = pdm1.get_homing_offset(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    print ("Limb 1 offsets: ")
    print ("Joint 1: %s" %(offsets[0]))
    print ("Joint 2: %s" %(offsets[1]))
    print ("Joint 3: %s" %(offsets[2]))
    print()

    print ("Limb 2 offsets: ")
    print ("Joint 4: %s" %(offsets[3]))
    print ("Joint 5: %s" %(offsets[4]))
    print ("Joint 6: %s" %(offsets[5]))
    print()

    print ("Limb 3 offsets: ")
    print ("Joint 7: %s" %(offsets[6]))
    print ("Joint 8: %s" %(offsets[7]))
    print ("Joint 9: %s" %(offsets[8]))
    print()

    print ("Limb 4 offsets: ")
    print ("Joint 10: %s" %(offsets[9]))
    print ("Joint 11: %s" %(offsets[10]))
    print ("Joint 12: %s" %(offsets[11]))
    print()
    print()

    # write to log file
    for i in range(1,13):
        filename = "homing_logs/joint_" + str(i) + ".txt"
        log = open(filename, "a+")
        now = datetime.datetime.now()
        data_entry = now.strftime("%Y-%m-%d %H:%M") + ", current, " + str(offsets[i-1]) + "\n"
        log.write(data_entry)
        log.close()

def read_positions():
    """
    Prints out the current position of each joint in radians. Can be used as a sanity check to make sure offsets are
    correct.
    """

    D1 = pdm1.get_bulk_status((1, 'present_position', 'present_velocity'),
                              (2, 'present_position', 'present_velocity'),
                              (3, 'present_position', 'present_velocity'),
                              (4, 'present_position', 'present_velocity'),
                              (5, 'present_position', 'present_velocity'),
                              (6, 'present_position', 'present_velocity'),
                              (7, 'present_position', 'present_velocity'),
                              (8, 'present_position', 'present_velocity'),
                              (9, 'present_position', 'present_velocity'),
                              (10, 'present_position', 'present_velocity'),
                              (11, 'present_position', 'present_velocity'),
                              (12, 'present_position', 'present_velocity'))
    print()
    print("Limb 1 positions: ")
    print("Joint 1: \t{}".format((D1[0][0])*ENC2RAD))
    print("Joint 2: \t{}".format(-(D1[1][0])*ENC2RAD))
    print("Joint 3: \t{}".format((D1[2][0] + D1[1][0])*ENC2RAD))
    print()
    print("Limb 2 positions: ")
    print("Joint 4: \t{}".format((D1[3][0])*ENC2RAD))
    print("Joint 5: \t{}".format(-(D1[4][0])*ENC2RAD))
    print("Joint 6: \t{}".format((D1[5][0] + D1[4][0])*ENC2RAD))
    print()
    print("Limb 3 positions: ")
    print("Joint 7: \t{}".format((D1[6][0])*ENC2RAD))
    print("Joint 8: \t{}".format(-(D1[7][0])*ENC2RAD))
    print("Joint 9: \t{}".format((D1[8][0] + D1[7][0])*ENC2RAD))
    print()
    print("Limb 4 positions: ")
    print("Joint 10: \t{}".format((D1[9][0])*ENC2RAD))
    print("Joint 11: \t{}".format(-(D1[10][0])*ENC2RAD))
    print("Joint 12: \t{}".format((D1[11][0] + D1[10][0])*ENC2RAD))
    print()
    print()


def home_hip_yaw_joint(joint):
    """
    Used to calibrate hip yaw joints, the nominal position is 0 degrees

    :param joint: joint id to be calibrated
    """

    raw_encoder_reading = 0.0
    for i in range(10):
        raw_encoder_reading += pdm1.get_present_position(joint)[0]/10.0
        time.sleep(0.01)

    current_offset = pdm1.get_homing_offset(joint)[0]

    new_offset = int(current_offset - raw_encoder_reading)
    if new_offset > 2**17:
        new_offset += - 2**18

    if new_offset > 2.0*np.pi*RAD2ENC:
        new_offset -= 2.0*np.pi*RAD2ENC
    elif new_offset < -2.0*np.pi*RAD2ENC:
        new_offset += 2.0*np.pi*RAD2ENC

    diff = new_offset - current_offset
    print("Current Offset: %s" %(current_offset))
    print("New Offset: %s" %(new_offset))
    print("Difference: %s" %(diff))
    print()
    print ("Do you want to use this new offset?")
    home = input("y/n: ")
    if home == "y":
        pdm1.set_homing_offset((joint, new_offset))
        pdm1.save_config(joint)

        # write to log file
        filename = "homing_logs/joint_" + str(joint) + ".txt"
        log = open(filename, "a+")
        now = datetime.datetime.now()
        data_entry = now.strftime("%Y-%m-%d %H:%M") + ", calibrated, " + str(new_offset) + "\n"
        log.write(data_entry)
        log.close()


def home_hip_pitch_joint(joint):
    """
    Used to calibrate hip pitch joints, the nominal position is -60 degrees

    :param joint: joint id to be calibrated
    """

    raw_encoder_reading = 0.0
    for i in range(10):
        raw_encoder_reading += pdm1.get_present_position(joint)[0]/10.0
        time.sleep(0.01)

    nominal = -60.0*np.pi/180.0*RAD2ENC
    current_offset = pdm1.get_homing_offset(joint)[0]

    new_offset = int(nominal + current_offset - raw_encoder_reading)
    if new_offset > 2**17:
        new_offset += - 2**18

    if new_offset > 2.0*np.pi*RAD2ENC:
        new_offset -= 2.0*np.pi*RAD2ENC
    elif new_offset < -2.0*np.pi*RAD2ENC:
        new_offset += 2.0*np.pi*RAD2ENC

    diff = new_offset - current_offset
    print("Current Offset: %s" %(current_offset))
    print("New Offset: %s" %(new_offset))
    print("Difference: %s" %(diff))
    print()
    print ("Do you want to use this new offset?")
    home = input("y/n: ")
    if home == "y":
        pdm1.set_homing_offset((joint, new_offset))
        pdm1.save_config(joint)

        # write to log file
        filename = "homing_logs/joint_" + str(joint) + ".txt"
        log = open(filename, "a+")
        now = datetime.datetime.now()
        data_entry = now.strftime("%Y-%m-%d %H:%M") + ", calibrated, " + str(new_offset) + "\n"
        log.write(data_entry)
        log.close()


def home_knee_pitch_joint(hip_joint, knee_joint):
    """
    Used to calibrate knee pitch joints, the nominal position is 0 degrees

    :param hip_joint: hip pitch id connected to the desired knee joint id
    :param knee_joint: joint id to be calibrated
    """

    raw_encoder_hip = 0.0
    raw_encoder_knee = 0.0
    for i in range(10):
        raw_encoder_hip += pdm1.get_bulk_status((hip_joint, 'present_position', 'present_velocity'))[0][0]/10.0
        raw_encoder_knee += pdm1.get_bulk_status((knee_joint, 'present_position', 'present_velocity'))[0][0]/10.0
        time.sleep(0.01)

    current_offset = pdm1.get_homing_offset(knee_joint)[0]
    new_offset = current_offset - raw_encoder_hip - raw_encoder_knee

    if new_offset > 2**17:
        new_offset += - 2**18

    if new_offset > 2.0*np.pi*RAD2ENC:
        new_offset -= 2.0*np.pi*RAD2ENC
    elif new_offset < -2.0*np.pi*RAD2ENC:
        new_offset += 2.0*np.pi*RAD2ENC

    diff = new_offset - current_offset
    print("Current Offset: %s" %(current_offset))
    print("New Offset: %s" %(new_offset))
    print("Difference: %s" %(diff))
    print()
    print ("Do you want to use this new offset?")
    home = input("y/n: ")
    if home == "y":
        pdm1.set_homing_offset((knee_joint, new_offset))
        pdm1.save_config(knee_joint)

        # write to log file
        filename = "homing_logs/joint_" + str(knee_joint) + ".txt"
        log = open(filename, "a+")
        now = datetime.datetime.now()
        data_entry = now.strftime("%Y-%m-%d %H:%M") + ", calibrated, " + str(new_offset) + "\n"
        log.write(data_entry)
        log.close()




if __name__ == '__main__':
    main()



