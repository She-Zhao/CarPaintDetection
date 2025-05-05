# from snap7 import client

# my_plc = client.Client()
# my_plc.connect("192.168.0.1", 0, 1)
# print(my_plc.get_connected())

# my_plc.disconnect()
# my_plc.destroy()


import numpy as np
import time
from snap7 import client, util, types
import socket


# -*-coding:utf-8 -*-
import screeninfo
import keyboard
# camera
import datetime
# camera end
import numpy as np
import os
import sys
import socket
from pypylon import pylon
import cv2
import time
import paramiko # 用于调用scp命令
from scp import SCPClient
from concurrent.futures import ThreadPoolExecutor
import torch


# global var
# DB10000
db_number = 10000
#TODO
min_start = 0
max_start = 310
# real/dint = 4; int = 2
end_type_size = 4

mode_index = [198, 154]      # mode_index = [program, plc]
veichel_state_index = 156
point_index = [200, 158]     # point_index = [expected, completed]
expected_pose_index = [286, 290, 294, 298, 302, 306, 310]
completed_pose_index = [66, 70, 74, 78, 82, 86, 90]
#TODO

class pose:
    def __init__(self):
        self.index = 0

        self.x = 0
        self.y = 0
        self.z = 0

        self.q0 = 0
        self.q1 = 0
        self.q2 = 0
        self.q3 = 0

class ProfinetCommunication():
    def __init__(self, ip, rack, slot):
        IP = ip
        RACK = rack
        SLOT = slot
        self.plc = client.Client()
        print("Connecting to PLC...")
        self.plc.connect(IP, RACK, SLOT)

        if self.plc.get_connected():
            print("Successfully connected to PLC!")
        else:
            print("Failed to connect to PLC! Please try again!")
            exit()

    def read_plc(self):
        byte_arrays = self.plc.read_area(types.Areas.DB, db_number, min_start, max_start+end_type_size)
        plc_mode = util.get_int(byte_arrays, mode_index[1])
        veichel_state = util.get_int(byte_arrays, veichel_state_index)
        # expected_point = util.get_int(byte_arrays, point_index[0])

        completed_pose = pose()
        completed_pose.index = util.get_int(byte_arrays, point_index[1])
        completed_pose.x  = util.get_real(byte_arrays, completed_pose_index[0])
        completed_pose.y  = util.get_real(byte_arrays, completed_pose_index[1])
        completed_pose.z  = util.get_real(byte_arrays, completed_pose_index[2])
        completed_pose.q0 = util.get_real(byte_arrays, completed_pose_index[3])
        completed_pose.q1 = util.get_real(byte_arrays, completed_pose_index[4])
        completed_pose.q2 = util.get_real(byte_arrays, completed_pose_index[5])
        completed_pose.q3 = util.get_real(byte_arrays, completed_pose_index[6])

        return plc_mode, veichel_state, completed_pose

    def write_plc(self, program_mode, expected_pose):
        byte_arrays = self.plc.read_area(types.Areas.DB, db_number, min_start, max_start+end_type_size)
        # program_mode
        util.set_int(byte_arrays, mode_index[0], program_mode)
        # expected_point
        util.set_int(byte_arrays, point_index[0], expected_pose.index)
        # expected_pose
        util.set_real(byte_arrays, expected_pose_index[0], expected_pose.x)
        util.set_real(byte_arrays, expected_pose_index[1], expected_pose.y)
        util.set_real(byte_arrays, expected_pose_index[2], expected_pose.z)
        util.set_real(byte_arrays, expected_pose_index[3], expected_pose.q0)
        util.set_real(byte_arrays, expected_pose_index[4], expected_pose.q1)
        util.set_real(byte_arrays, expected_pose_index[5], expected_pose.q2)
        util.set_real(byte_arrays, expected_pose_index[6], expected_pose.q3)

        self.plc.write_area(types.Areas.DB, db_number, min_start, byte_arrays)

    def close_connect(self):
        self.plc.disconnect()
        self.plc.destroy()

if __name__ == '__main__':
    main_point = np.loadtxt('/home/nvidia/demo/host/python_S7/main_point.txt')
    len_main = len(main_point)
    expected_point = {}
    for a in range(len_main):
        expected_point[a] = pose()
        expected_point[a].index = a
        expected_point[a].x  = main_point[a][0]
        expected_point[a].y  = main_point[a][1]
        expected_point[a].z  = main_point[a][2]
        expected_point[a].q0 = main_point[a][3]
        expected_point[a].q1 = main_point[a][4]
        expected_point[a].q2 = main_point[a][5]
        expected_point[a].q3 = main_point[a][6]
        expected_point[a].cf1 = main_point[a][7]
        expected_point[a].cf4 = main_point[a][8]
        expected_point[a].cf6 = main_point[a][9]
        expected_point[a].cfx = main_point[a][10]

    Home_pose = pose()
    # Home_pose = expected_point[0]

    #TODO
    plc_IP = "10.18.18.31"
    # S7-1200 0/0; S7-1500 0/1
    plc_rack = 0
    plc_slot = 1
    #TODO

    program_mode = 3
    plc_mode = 0
    veichel_state = 0
    completed_pose = Home_pose

    plc_S7_1500 = ProfinetCommunication(plc_IP, plc_rack, plc_slot)
    plc_mode, veichel_state, completed_pose = plc_S7_1500.read_plc()
    print("pose: ", completed_pose.x, completed_pose.y, completed_pose.z, completed_pose.q0, completed_pose.q1, completed_pose.q2, completed_pose.q3)

    # plc_S7_1500.write_plc(program_mode, expected_point[1])
    # time.sleep(3.0)


    plc_S7_1500.close_connect()
    print("PLC is disconnected")
