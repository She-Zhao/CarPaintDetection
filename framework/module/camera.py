import os
import pypylon.pylon as py
import cv2
import time
from pathlib import Path

class CameraControl :
    def __init__(self, exposure_time=8000, height=None, width=None,
                 max_frames=10, capture_callback=None):
        self.exposure_time = exposure_time
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.capture_callback = capture_callback
        self._camera_init()

    def _camera_init(self):
        try:
            tlf = py.TlFactory.GetInstance()
            di = py.DeviceInfo()
            di.SetDeviceClass("BaslerGigE")

            devs = tlf.EnumerateDevices([di,])      # 发现可用设备
            if not devs:
                raise RuntimeError("未检测到任何Basler GigE相机")

            # 根据实际设备数量创建相机数组
            num_cameras = len(devs)
            print(f"发现 {num_cameras} 台相机")

            self.cam_array = py.InstantCameraArray(num_cameras)
            self.img_buffers = {}

            # 绑定并初始化相机
            for idx, cam in enumerate(self.cam_array):

                # 关联物理设备
                cam.Attach(tlf.CreateDevice(devs[idx]))
                cam.Open()

                # 配置基础参数
                if self.exposure_time: cam.ExposureTime.SetValue(self.exposure_time)
                if self.height: cam.Height.Value = self.height
                if self.width: cam.Width.Value = self.width

                # 生成唯一标识
                serial = int(cam.DeviceInfo.GetSerialNumber())
                cam.SetCameraContext(serial)            # 使用序列号作为上下文标识，传入的参数必须是int型
                self.img_buffers[serial] = []

        except py.GenericException as e:
            print(f"Pylon错误: {e}")

    def start_grabbing(self):
        try:
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
                        print(f">>>>>相机{serial}，照片数量为{len(self.img_buffers[serial])}")

                        if self.capture_callback:
                            self.capture_callback(serial)  # 触发回调传递序列号，向主机发送切换请求

                        frame_counts[serial] += 1
                        if all(cnt >= self.max_frames for cnt in frame_counts.values()):    # 拍摄10张照片
                            break

                finally:
                    res.Release()
            self._save_images()

        except py.TimeoutException:
            print("采集超时，请检查相机连接")
        finally:
            self.cam_array.StopGrabbing()

    def _save_images(self):
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
                cv2.imwrite(os.path.join(save_dir, f"{img_idx:04d}.png"), img)
            print(f"相机 {serial} 已保存 {len(imgs)} 张图像")
            self.img_buffers[serial] = []
