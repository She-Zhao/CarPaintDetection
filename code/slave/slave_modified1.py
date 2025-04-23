# -*- coding: utf-8 -*-
import argparse
import logging
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from pypylon import pylon

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_DELAY_MS = 60
DEFAULT_PORT = 4096
HOST_IP = '10.18.18.11'
IMAGE_NAMES = ["zhj", "gc0", "gc1", "gc2", "gc3", "gc4", "sin0", "sin1", "sin2", "sin3"]
REQUIRED_CAPTURES = 10
SOCKET_BUFFER_SIZE = 1024
RESPONSE_MESSAGE = "switch_pattern"
OUTPUT_DIR = Path(__file__).parent / "output/Camera"
FLAG_FILE = Path(__file__).parent / "FLAG.txt"


@contextmanager
def camera_session():
    """上下文管理器自动处理相机资源"""
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise EnvironmentError("No GigE devices found")

    target_device = next((dev for dev in devices if dev.GetIpAddress().endswith('1')), None)
    if not target_device:
        raise EnvironmentError("Target camera (IP ending with .1) not found")

    camera = pylon.InstantCamera(tl_factory.CreateDevice(target_device))
    camera.Open()
    try:
        yield camera
    finally:
        camera.Close()


def setup_socket_server(host: str, port: int) -> socket.socket:
    """初始化并绑定Socket服务器"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    logger.info(f"Server listening on {host}:{port}")
    return server_socket


def save_images(images: list, pos_index: int) -> Path:
    """保存图像到指定位置，返回生成目录路径"""
    output_folder = OUTPUT_DIR / f"pos{pos_index}"
    while output_folder.exists():
        pos_index += 1
        output_folder = OUTPUT_DIR / f"pos{pos_index}"
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for idx, img in enumerate(images[1:REQUIRED_CAPTURES+1], 1):  # 跳过第一张
        img_name = IMAGE_NAMES[idx] if idx < len(IMAGE_NAMES) else f"unknown_{idx}"
        cv2.imwrite(str(output_folder / f"{img_name}.png"), img)
    
    return output_folder


def handle_capture_cycle(camera, connection, delay_ms: int) -> int:
    """处理一次完整的采集周期，返回当前点位计数"""
    img_buffer = []
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    try:
        while camera.IsGrabbing() and len(img_buffer) < REQUIRED_CAPTURES:
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                img_buffer.append(grab_result.Array)
                connection.sendall(RESPONSE_MESSAGE.encode('utf-8'))
                
                # 非阻塞延迟
                start_time = time.perf_counter()
                while (time.perf_counter() - start_time) * 1000 < delay_ms:
                    if connection.recv(SOCKET_BUFFER_SIZE, socket.MSG_DONTWAIT):
                        break  # 如果收到新指令则提前退出延迟
                
                grab_result.Release()
    finally:
        camera.StopGrabbing()
    
    return len(img_buffer), img_buffer  # 同时返回数量和实际数据


def main():
    parser = argparse.ArgumentParser(description='相机采集控制系统')
    parser.add_argument('--delay', type=int, default=DEFAULT_DELAY_MS, help='采集间隔(毫秒)')
    parser.add_argument('--camera-num', type=int, default=5, dest='camera_num', help='总采集点位数量')
    args = parser.parse_args()

    pos_index = 1
    with setup_socket_server(HOST_IP, DEFAULT_PORT) as server_socket:
        while True:
            conn, addr = server_socket.accept()
            logger.info(f"Connected by {addr}")
            
            with conn, camera_session() as cam:
                while True:
                    data = conn.recv(SOCKET_BUFFER_SIZE)
                    if not data:
                        logger.warning("Connection closed by client")
                        break
                    
                    command = data.decode('utf-8').strip()
                    logger.info(f"Received command: {command}")
                    
                    if command == "capture_order":
                        captured_count, captured_images = handle_capture_cycle(cam, conn, args.delay)
                        
                        if captured_count >= REQUIRED_CAPTURES:
                            save_images(captured_images, pos_index)
                            pos_index += 1
                            
                            with FLAG_FILE.open('w') as f:
                                f.write(f'CAMERA_FLAG={pos_index}\n')
                            logger.info(f"Updated camera flag to {pos_index}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Critical error occurred: {str(e)}", exc_info=True)