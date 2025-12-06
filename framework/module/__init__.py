# module/__init__.py
from .camera import CameraControl
from .socket import SocketServer
from .host import HostControl
from .model_config import ModelConfigManager

__all__ = ['CameraControl', 'SocketServer', 'HostControl', 'ModelConfigManager']  # 明确暴露的接口
