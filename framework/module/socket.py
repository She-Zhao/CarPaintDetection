"""
文件: socket.py
功能: 实现带端口冲突自动重试的TCP服务端，提供安全收发机制
依赖: socket, typing.Optional, typing.Tuple

典型用法:
>>> with SocketServer(host='127.0.0.1', base_port=5000) as server:
...     data = server.safe_receive()  # 接收UTF-8编码数据
...     server.safe_send("ACK")       # 发送UTF-8字符串
"""
# -*-coding:utf-8 -*-
import socket
from typing import Optional, Tuple

class SocketServer:
    """智能TCP服务端，支持端口冲突自动重试和安全连接管理
    
    主要功能层级:
    ├─ 初始化配置: 设置监听参数 (__init__)
    ├─ 连接管理:
    │    ├─ 建立监听连接 (connect)
    │    └─ 安全关闭连接 (close)
    ├─ 数据传输:
    │    ├─ 安全接收数据 (safe_receive)
    │    └─ 安全发送数据 (safe_send)
    └─ 上下文管理:
         ├─ 进入上下文自动连接 (__enter__)
         └─ 退出上下文自动清理 (__exit__)
    
    核心特性:
    - 端口占用时自动递增端口号重试 (base_port -> base_port+1...)
    - 收发数据时自动检查连接状态
    - 通过上下文管理器保证异常安全
    """
    def __init__(self, host: str, base_port: int, retries: int = 3):
        """初始化服务端配置
        Args:
            host: 监听地址，'0.0.0.0'表示接受任意来源连接
            base_port: 起始监听端口，冲突时按+1递增尝试
            retries: 最大端口重试次数，默认3次（即尝试base_port到base_port+2）
        """
        self.host = host
        self.base_port = base_port
        self.retries = retries
        self.server_socket: Optional[socket.socket] = None
        self.connection: Optional[socket.socket] = None
        self.connected = False

    def __enter__(self):
        """实现上下文管理器入口，自动调用connect()"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """实现上下文管理器退出，自动调用close()"""
        self.close()

    def connect(self) -> Tuple[socket.socket, socket.socket]:
        """建立TCP监听并接受客户端连接
        Returns:
            Tuple: (server_socket, client_socket) 元组
            
        Raises:
            RuntimeError: 所有重试端口均被占用时抛出
            OSError: 底层socket操作异常时抛出
            
        示例流程:
        1. 尝试绑定 base_port
        2. 失败则递增端口直至成功或达到retries限制
        3. 开启监听并阻塞等待客户端连接
        """
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
        """安全接收UTF-8编码数据
        Args:
            bufsize: 单次接收缓冲区大小（字节），默认1024B
            
        Returns:
            str: 解码后的UTF-8字符串
            
        Raises:
            ConnectionError: 未建立连接时调用
            ConnectionAbortedError: 连接被远端关闭
        """
        if not self.connected:
            raise ConnectionError("连接未建立")
            
        data = self.connection.recv(bufsize)
        if not data:
            self.connected = False
            raise ConnectionAbortedError("连接被主机关闭")
            
        return data.decode('utf-8')

    def safe_send(self, message: str) -> None:
        """安全发送UTF-8编码数据
        Args:
            message: 待发送字符串，自动进行UTF-8编码
            
        Raises:
            ConnectionError: 未建立连接或连接中断
        """
        if not self.connected:
            raise ConnectionError("连接未建立")
            
        self.connection.sendall(message.encode('utf-8'))

    def close(self) -> None:
        """安全关闭连接，执行顺序:
        1. 关闭客户端socket的读写通道
        2. 关闭客户端socket对象
        3. 关闭服务端socket
        4. 重置连接状态标志
        """
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
        