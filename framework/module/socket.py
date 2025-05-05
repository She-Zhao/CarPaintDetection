# -*-coding:utf-8 -*-
import socket
from typing import Optional, Tuple

class SocketServer:
    """可自动重试的 TCP 服务端封装类"""
    def __init__(self, host: str, base_port: int, retries: int = 3):
        """
        Args:
            host: 监听地址
            base_port: 起始端口号
            retries: 端口冲突时的重试次数
        """
        self.host = host
        self.base_port = base_port
        self.retries = retries
        self.server_socket: Optional[socket.socket] = None
        self.connection: Optional[socket.socket] = None
        self.connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self) -> Tuple[socket.socket, socket.socket]:
        """建立 socket 连接"""
        current_port = self.base_port
        for attempt in range(self.retries):
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, current_port))
                self.server_socket.listen(10)
                print(f"✅ 成功监听 {self.host}:{current_port}")
                
                self.connection, address = self.server_socket.accept()
                print(f"🔗 已接受来自 {address} 的连接")
                self.connected = True
                return self.server_socket, self.connection
                
            except OSError as e:
                if attempt < self.retries - 1:
                    print(f"⚠️ 端口 {current_port} 被占用，尝试 {current_port+1}...")
                    current_port += 1
                else:
                    raise RuntimeError(f"无法绑定端口，最后尝试端口: {current_port}") from e

    def safe_receive(self, bufsize: int = 1024) -> str:
        """带连接状态检查的数据接收方法"""
        if not self.connected:
            raise ConnectionError("连接未建立")
            
        data = self.connection.recv(bufsize)
        if not data:
            self.connected = False
            raise ConnectionAbortedError("连接被主机关闭")
            
        return data.decode('utf-8')

    def safe_send(self, message: str) -> None:
        """带连接状态检查的数据发送方法"""
        if not self.connected:
            raise ConnectionError("连接未建立")
            
        self.connection.sendall(message.encode('utf-8'))

    def close(self) -> None:
        """安全关闭连接"""
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            except OSError:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass
        self.connected = False
        print("🔒 Socket 连接已安全关闭")