"""
save_robot_pose.py - 保存机械臂位姿工具

该模块负责保存机械臂位姿到txt中，即原来的collect_imgs.py

功能说明:
    - 调整机械臂到指定位置
    - 运行save_robot_pose.py

最后一次修改：2025/05/05
"""

import numpy as np
import time
from snap7 import client, util, types
import socket
import os

db_number = 10000

min_start = 0
max_start = 326  # 理论上不需要从0读取整个需要的DB块数据，实际测试不从0读取会发生错误

# real/dint = 4; int = 2
end_type_size = 4

mode_index = [198, 154]      # mode_index = [program, plc]
veichel_state_index = 156
point_index = [200, 158]     # point_index = [expected, completed]
expected_pose_index = [286, 290, 294, 298, 302, 306, 310, 314, 318, 322, 326]
completed_pose_index = [66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106]

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
    # take and save photos
    # folder_path = "/home/nvidia/demo/host/patterns/16"
    # host = Host(folder_path)
    
    # # client_socket_1,client_socket_2,client_socket_3,client_socket_4 = host.Socket_init()     #初始化Socket通信
    # host.Socket_init()
    # # host.Projected_init()

    # host.Take_once()


    # get and save robot paras

    plc_IP = "10.18.18.31"
    # S7-1200 0/0; S7-1500 0/1
    plc_rack = 0
    plc_slot = 1

    program_mode = 3
    plc_mode = 0
    veichel_state = 0

    plc_S7_1500 = ProfinetCommunication(plc_IP, plc_rack, plc_slot)
    plc_mode, veichel_state, completed_pose = plc_S7_1500.read_plc()

    with open('robot.txt', 'a') as file: 
        file.write(str(completed_pose.x) + ' ' + str(completed_pose.y) + ' ' + str(completed_pose.z) + ' ' 
                   + str(completed_pose.q0) + ' ' + str(completed_pose.q1) + ' ' + str(completed_pose.q2) + ' ' + str(completed_pose.q3) + '\n')

    plc_S7_1500.close_connect()