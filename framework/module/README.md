# Module 目录说明

本目录包含系统核心功能模块的实现，以下是各文件的详细说明：

## 📁 模块文件清单

| 文件 | 功能描述 | 主要接口 |
|------|----------|----------|
| [`camera.py`](./camera.py) | Basler相机控制模块 | `CameraControl` 类<br>（多相机同步采集/图像存储） |
| [`host.py`](./host.py) | 主机控制核心模块 | `HostControl` 类<br>（投影控制/从机通信） |
| [`socket.py`](./socket.py) | 增强型Socket通信模块 | `SocketServer` 类<br>（自动重试/安全收发） |
