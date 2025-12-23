import struct
import json
import numpy as np
import cv2
import socket

class DataProtocol:
    """
    通信协议定义:
    [包头 12 bytes] = [JSON长度 (int)] + [图片长度 (int)] + [保留字段 (int)]
    [包体]          = [JSON bytes] + [图片 bytes]
    """
    
    @staticmethod
    def pack_data(image: np.ndarray, defect_data: list) -> bytes:
        """
        打包数据：将 图像(numpy) 和 缺陷列表(list) 打包成字节流
        """
        # 1. 处理 JSON
        # ensure_ascii=False 允许传输中文，separators 去除空格减小体积
        json_str = json.dumps(defect_data, ensure_ascii=False, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        
        # 2. 处理图像 (编码为 PNG)
        # 工业检测推荐 PNG (无损)，如果追求极致速度可用 JPG
        success, img_encoded = cv2.imencode('.png', image)
        if not success:
            raise ValueError("图像编码失败")
        img_bytes = img_encoded.tobytes()
        
        # 3. 构建包头 (Big-Endian, 3个 unsigned int)
        # >III: 大端序, int, int, int (每个占4字节，共12字节)
        header = struct.pack('>III', len(json_bytes), len(img_bytes), 0)
        
        # 4. 拼接返回
        return header + json_bytes + img_bytes

    @staticmethod
    def recv_exact(sock: socket.socket, length: int) -> bytes:
        """
        辅助函数：从 socket 读取指定长度的字节，处理 TCP 拆包/粘包问题
        """
        data = b''
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                # 连接中断
                raise ConnectionError("对方关闭了连接")
            data += packet
        return data

    @staticmethod
    def unpack_data(sock: socket.socket):
        """
        解包数据：从 socket 读取并解析，返回 (image, json_data)
        """
        # 1. 读取包头 (12字节)
        header_data = DataProtocol.recv_exact(sock, 12)
        json_len, img_len, _ = struct.unpack('>III', header_data)
        
        # 2. 读取 JSON
        json_bytes = DataProtocol.recv_exact(sock, json_len)
        defect_data = json.loads(json_bytes.decode('utf-8'))
        
        # 3. 读取图像
        img_bytes = DataProtocol.recv_exact(sock, img_len)
        img_np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        
        return image, defect_data