# -*-coding:utf-8 -*-
import numpy as np
import time
# from snap7 import client, util, types
import screeninfo
import keyboard

import socket
from pypylon import pylon
import cv2
import paramiko # 用于调用scp命令
from scp import SCPClient
from concurrent.futures import ThreadPoolExecutor
import torch
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
framework_root = current_file.parent.parent
sys.path.insert(0, str(framework_root))

from module import Host


# global var
# DB10000
db_number = 10000
#TODO
min_start = 0
max_start = 326
# real/dint = 4; int = 2
end_type_size = 4

mode_index = [198, 154]      # mode_index = [program, plc]
veichel_state_index = 156
point_index = [200, 158]     # point_index = [expected, completed]
expected_pose_index = [286, 290, 294, 298, 302, 306, 310, 314, 318, 322, 326]
completed_pose_index = [66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106]
#TODO
Home_index = 1
# min_start = 0
# max_start = 30
# end_type_size = 4

# mode_index = [0, 0]
# veichel_state_index = 2
# point_index = [4, 4]
# expected_pose_index = [6, 10, 14, 18, 22, 26,30]
# completed_pose_index = [6, 10, 14, 18, 22, 26,30]

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

        self.cf1 = 0
        self.cf4 = 0
        self.cf6 = 0
        self.cfx = 0

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
        completed_pose.cf1 = util.get_real(byte_arrays, completed_pose_index[7])
        completed_pose.cf4 = util.get_real(byte_arrays, completed_pose_index[8])
        completed_pose.cf6 = util.get_real(byte_arrays, completed_pose_index[9])
        completed_pose.cfx = util.get_real(byte_arrays, completed_pose_index[10])

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
        util.set_real(byte_arrays, expected_pose_index[7], expected_pose.cf1)
        util.set_real(byte_arrays, expected_pose_index[8], expected_pose.cf4)
        util.set_real(byte_arrays, expected_pose_index[9], expected_pose.cf6)
        util.set_real(byte_arrays, expected_pose_index[10], expected_pose.cfx)

        self.plc.write_area(types.Areas.DB, db_number, min_start, byte_arrays)

    def close_connect(self):
        self.plc.disconnect()
        self.plc.destroy()


if __name__ == '__main__':
    folder_path = r'/home/nvidia/demo/host/patterns/16'
    host = Host(folder_path)
    host.Socket_init()
    host.Projected_init()
    
    main_point = np.loadtxt('/home/nvidia/demo/host/python_S7/main_point.txt')
    len_main = len(main_point)
    expected_point = {}
    for a in range(len_main):
        b = a
        expected_point[b] = pose()
        expected_point[b].index = a + 1
        expected_point[b].x  = main_point[a][0]
        expected_point[b].y  = main_point[a][1]
        expected_point[b].z  = main_point[a][2]
        expected_point[b].q0 = main_point[a][3]
        expected_point[b].q1 = main_point[a][4]
        expected_point[b].q2 = main_point[a][5]
        expected_point[b].q3 = main_point[a][6]
        expected_point[b].cf1 = main_point[a][7]
        expected_point[b].cf4 = main_point[a][8]
        expected_point[b].cf6 = main_point[a][9]
        expected_point[b].cfx = main_point[a][10]

    Home_pose = pose()
    Home_pose = expected_point[0]

    #TODO
    plc_IP = "10.18.18.31"
    # S7-1200 0/0; S7-1500 0/1
    plc_rack = 0
    plc_slot = 1
    #TODO

    program_mode = 0
    plc_mode = 0
    veichel_state = 0
    completed_pose = Home_pose

    plc_S7_1500 = ProfinetCommunication(plc_IP, plc_rack, plc_slot)

    while True:
        plc_mode, veichel_state, completed_pose = plc_S7_1500.read_plc()
        print("Present PLC mode [" + str(plc_mode) + "]")
        if plc_mode == 3: #TODO 3
            program_mode = 3
            print("PLC has been ready!")
            plc_S7_1500.write_plc(program_mode, Home_pose)
            while True:
                plc_mode, veichel_state, completed_pose = plc_S7_1500.read_plc()
                if completed_pose.index == Home_index:
                    print("Robot in home!")
                    break
            break
        time.sleep(0.5)

    
    
    for i in range(1, len_main):
        plc_S7_1500.write_plc(program_mode, expected_point[i])
        print("Point[" + str(i) + "] has been sent to the robot!")
        
        while True:
            plc_mode, veichel_state, completed_pose = plc_S7_1500.read_plc()
            print("Index: " + str(completed_pose.index) + " PLC_Mode: " + str(plc_mode) + "  Present [X: " + str(completed_pose.x) + "; Y: " + str(completed_pose.y) + "; Z: " + str(completed_pose.z) + "]")
            if completed_pose.index == i + 1: #TODO i
                # time.sleep(1.0)
                print("Arrived at point [" + str(i) + "]!")

                host.Take_photo()

                break
            time.sleep(0.1)

    # plc_mode, veichel_state, completed_pose = plc_S7_1500.read_plc()
    # print("plc_mode: " + str(plc_mode) + "; veichel_state: " + str(veichel_state) + "; completed_pose: " + str(completed_pose.index))
    # print("pose: ", completed_pose.x, completed_pose.y, completed_pose.z)
    plc_S7_1500.write_plc(program_mode, Home_pose)
    print("Resetting...!")
    host.Disconnect()
    while True:
        plc_mode, veichel_state, completed_pose = plc_S7_1500.read_plc()
        if completed_pose.index == Home_index:
            print("Reset Done!")
            program_mode = 0
            plc_S7_1500.write_plc(program_mode, Home_pose)
            break
    plc_S7_1500.close_connect()
    print("PLC is disconnected")