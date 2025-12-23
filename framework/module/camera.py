"""
文件: camera.py
功能: 管理多台Basler GigE相机，实现图像采集与存储
依赖: pypylon.pylon, cv2, time, os, pathlib.Path

典型用法:
>>> from module.camera import CameraControl
>>> def callback(serial):
...     print(f"相机 {serial} 捕获到新帧")
>>> cam_ctl = CameraControl(exposure_time=8000, max_frames=10, capture_callback=callback)
>>> cam_ctl.start_grabbing()
"""

import os
import pypylon.pylon as py
import cv2
import time
from pathlib import Path

  # 存储图像的命名顺序

class CameraControl :
    """Basler相机控制器
    
    主要功能层级:
    ├─ 初始化配置: 自动检测相机并设置参数 (_camera_init)
    ├─ 采集控制: 启动/停止图像流，管理采集流程 (start_grabbing)
    └─ 存储管理: 按序列号和时间戳保存图像 (_save_images)
    
    属性:
        IMAGE_NAMES (list): 图像文件名前缀列表，按采集顺序命名
    """
    IMAGE_NAMES= ["zhj","gc0","gc1","gc2","gc3","gc4","sin0","sin1","sin2","sin3"]  # 作为类常量，所有实例共享一个

    def __init__(self, exposure_time=8000, height=None, width=None,
                 max_frames=10, capture_callback=None):
        """初始化相机控制器
        Args:
            exposure_time (int): 曝光时间（微秒），默认8000μs
            height (int): 图像高度像素值，None保持相机默认
            width (int): 图像宽度像素值，None保持相机默认 
            max_frames (int): 单次采集最大帧数，默认10帧
            capture_callback (function): 图像捕获回调函数，接收serial参数
        """        
        self.exposure_time = exposure_time
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.capture_callback = capture_callback
        self._camera_init()

    def _camera_init(self):
        """初始化相机硬件连接"""
        # [修改点1] 去掉 try...except，或者捕获后抛出。建议直接去掉，让错误暴露出来。
        tlf = py.TlFactory.GetInstance()
        di = py.DeviceInfo()
        di.SetDeviceClass("BaslerGigE")

        devs = tlf.EnumerateDevices([di,])
        if not devs:
            raise RuntimeError("未检测到任何Basler GigE相机")

        num_cameras = len(devs)
        print(f"发现 {num_cameras} 台相机")

        self.cam_array = py.InstantCameraArray(num_cameras)
        self.img_buffers = {}

        for idx, cam in enumerate(self.cam_array):
            cam.Attach(tlf.CreateDevice(devs[idx]))
            # 如果这里 Open 失败（比如被占用），程序会直接抛出异常，
            # 这样 client.py 就能捕获到，而不会带着坏掉的对象继续跑。
            cam.Open() 

            if self.exposure_time: cam.ExposureTime.SetValue(self.exposure_time)
            if self.height: cam.Height.Value = self.height
            if self.width: cam.Width.Value = self.width

            serial = int(cam.DeviceInfo.GetSerialNumber())
            cam.SetCameraContext(serial)
            self.img_buffers[serial] = []

    def start_grabbing(self):
        """启动多相机同步采集流程
        
        工作流程:
        1. 启动相机采集流
        2. 循环获取图像直到达到max_frames
        3. 触发回调并自动保存图像
        4. 超时5秒未收到图像则中断
        
        return:
            [相机1采集的一组图像，相机2采集的一组图像]
            
        Raises:
            py.TimeoutException: 图像采集超时时抛出
        """        
        try:
            # === [新增] 必须在开始采集前清空旧数据 ===
            for serial in self.img_buffers:
                self.img_buffers[serial] = [] # 清空列表
            # ========================================
            
            self.cam_array.StartGrabbing(py.GrabStrategy_LatestImageOnly)
            # 两个相机从调用StartGrabbing到可以拍照需要时间，不加延时的话其中一个相机准备好了会先拍照导致时许对不上
            time.sleep(0.05)
            grab_timeout = 5000  # 超时时间5秒
            frame_counts = {serial:0 for serial in self.img_buffers.keys()}

            while True:
                # 获取图像结果
                res = self.cam_array.RetrieveResult(grab_timeout, py.TimeoutHandling_ThrowException)
                try:
                    if res.GrabSucceeded():
                        serial = res.GetCameraContext()
                        self.img_buffers[serial].append(res.Array)
                        # print(f">>>>>相机{serial}，照片数量为{len(self.img_buffers[serial])}")

                        if self.capture_callback:
                            self.capture_callback(serial)  # 触发回调传递序列号，向主机发送切换请求

                        frame_counts[serial] += 1
                        if all(cnt >= self.max_frames for cnt in frame_counts.values()):    # 拍摄10张照片
                            break

                finally:
                    res.Release()
            # self._save_images()

        except py.TimeoutException:
            print("采集超时，请检查相机连接")
        finally:
            self.cam_array.StopGrabbing()
            
        return list(self.img_buffers.values())

    def _save_images(self):
        """保存缓冲图像到output目录
        
        存储路径结构:
        framework/output/[相机序列号]/pos[序号]/[IMAGE_NAMES].png
        例如: output/123456/pos0/zhj.png
        """        
        for serial, imgs in self.img_buffers.items():
            base_dir = Path(__file__).resolve().parent.parent            # 当前文件的上上一级，framework文件夹 
            root_dir = os.path.join(base_dir, "output", str(serial))     # framework/output/相机序列号
            os.makedirs(root_dir, exist_ok=True)                         

            # 查找当前最大的 pos 索引
            existing_indices = []
            for name in os.listdir(root_dir):
                if name.startswith("pos"):
                    try:
                        index = int(name[3:])
                        existing_indices.append(index)
                    except ValueError:
                        continue
            next_idx = max(existing_indices) + 1 if existing_indices else 0

            # 创建新目录
            save_dir = os.path.join(root_dir, f"pos{next_idx}")
            os.makedirs(save_dir, exist_ok=True)

            # 保存图像
            for img_idx, img in enumerate(imgs):
                cv2.imwrite(os.path.join(save_dir, f"{self.IMAGE_NAMES[img_idx]}.png"), img)
            print(f"相机 {serial} 已保存 {len(imgs)} 张图像")
            self.img_buffers[serial] = []

    # [修改点2] 新增资源释放方法
    def release(self):
        """释放相机资源，关闭设备连接"""
        if hasattr(self, 'cam_array') and self.cam_array:
            try:
                # 如果正在采集，先停止
                if self.cam_array.IsGrabbing():
                    self.cam_array.StopGrabbing()
                
                # 关闭连接，释放硬件锁
                if self.cam_array.IsOpen():
                    self.cam_array.Close()
                
                # 解除绑定
                self.cam_array.DetachDevice()
                print("📷 相机资源已释放")
            except Exception as e:
                print(f"释放相机资源时出错: {e}")

    def __del__(self):
        """析构函数，作为最后一道防线"""
        self.release()