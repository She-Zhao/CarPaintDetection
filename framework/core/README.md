# Core 目录说明

本目录包含系统核心控制逻辑的实现，以下是各文件的详细说明：

## 📁 文件清单

| 文件 | 类型 | 功能描述 | 主要接口 |
|------|------|----------|----------|
| [`client.py`](./client.py) | 从机核心 | 相机采集控制与协议处理 | `CaptureTracker` 类<br>`create_camera_with_callback()` |
| [`server.py`](./server.py) | 主机核心 | 主控制流程与用户交互 | `main()` 入口函数 |
