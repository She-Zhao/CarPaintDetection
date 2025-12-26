# import pypylon.pylon as py
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os

# # NUM_CAMERAS = 10

# # setup demo environment with 10 cameras
# # os.environ["PYLON_CAMEMU"] = f"{NUM_CAMERAS}"

# tlf = py.TlFactory.GetInstance()

# # create a device filter for Pylon CamEmu devices
# di = py.DeviceInfo()
# di.SetDeviceClass("BaslerGigE")  # 千兆网相机

# # you could also set more device filters like e.g.:
# # these are combined as AND 
# # di.SetSerialNumber("2134234")

# devs = tlf.EnumerateDevices([di,])

# print(devs)
# cam_array = py.InstantCameraArray(len(devs))

# for idx, cam in enumerate(cam_array):
#     cam.Attach(tlf.CreateDevice(devs[idx]))

# cam_array.Open()

# # store a unique number for each camera to identify the incoming images
# for idx, cam in enumerate(cam_array):
#     camera_serial = cam.DeviceInfo.GetSerialNumber()
#     print(f"set context {idx} for camera {camera_serial}")
#     cam.SetCameraContext(idx)

# # set the exposure time for each camera
# for idx, cam in enumerate(cam_array):
#     camera_serial = cam.DeviceInfo.GetSerialNumber()
#     print(f"set Exposuretime {idx} for camera {camera_serial}")
#     cam.ExposureTime.SetValue(8000)


# # wait for all cameras to grab 10 frames
# frames_to_grab = 10
# # store last framecount in array
# frame_counts = [0]*len(devs)

# frame_counts

# cam_array.StartGrabbing()
# while True:
#     with cam_array.RetrieveResult(1000) as res:
#         if res.GrabSucceeded():
#             img_nr = res.ImageNumber
#             cam_id = res.GetCameraContext()
#             frame_counts[cam_id] = img_nr
#             print(f"cam #{cam_id}  image #{img_nr}")
            
#             # do something with the image ....
            
#             # check if all cameras have reached 100 images
#             if min(frame_counts) >= frames_to_grab:
#                 print( f"all cameras have acquired {frames_to_grab} frames")
#                 break
                
                
# cam_array.StopGrabbing()

# cam_array.Close()

# frame_counts


import pypylon.pylon as py
import cv2
import os
from datetime import datetime

class Camera:
    def __init__(self, exposure_time=8000, height=None, width=None, max_frames=10):
        self.exposure_time = exposure_time
        self.height = height
        self.width = width        
        self.max_frames = max_frames
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
                if self.exposure_time: cam.ExposureTime.SetValue(self.exposure_time)       # 曝光时间10ms
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
            grab_timeout = 5000  # 超时时间5秒
            frame_counts = {serial:0 for serial in self.img_buffers.keys()}

            while True:
                # 获取图像结果
                res = self.cam_array.RetrieveResult(grab_timeout, py.TimeoutHandling_ThrowException)
                try:
                    if res.GrabSucceeded():
                        serial = res.GetCameraContext()
                        self.img_buffers[serial].append(res.Array)

                        frame_counts[serial] += 1
                        if all(cnt >= self.max_frames for cnt in frame_counts.values()):
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
            save_dir = os.path.join(os.getcwd(), str(serial))
            os.makedirs(save_dir, exist_ok=True)

            for idx, img in enumerate(imgs):
                cv2.imwrite(os.path.join(save_dir, f"{idx:04d}.png"), img)
            print(f"相机 {serial} 已保存 {len(imgs)} 张图像")

if __name__ == "__main__":
    try:
        camera = Camera(exposure_time=8000, max_frames=10)  # 每台相机采集10帧
        camera.start_grabbing()  # 阻塞直到完成采集和保存
    except Exception as e:
        print(f"运行失败: {str(e)}")
        

# # 创建输出根目录
# output_root = os.path.join(os.path.expanduser("~"), "BaslerCapture")
# os.makedirs(output_root, exist_ok=True)

# # 初始化Pylon TLFactory
# tlf = py.TlFactory.GetInstance()

# # 配置设备过滤器
# di = py.DeviceInfo()
# di.SetDeviceClass("BaslerGigE")  # 指定千兆网相机

# try:
#     # 发现可用设备
#     devs = tlf.EnumerateDevices([di,])
#     if not devs:
#         raise RuntimeError("未检测到任何Basler GigE相机")

#     # 根据实际设备数量创建相机数组
#     num_cameras = len(devs)
#     print(f"发现 {num_cameras} 台相机")
    
#     cam_array = py.InstantCameraArray(num_cameras)

#     # 绑定并初始化相机
#     for idx, cam in enumerate(cam_array):
#         # 关联物理设备
#         cam.Attach(tlf.CreateDevice(devs[idx]))
        
#         # 打开相机连接
#         cam.Open()
        
#         # 配置基础参数
#         cam.ExposureTime.SetValue(8000)       # 曝光时间10ms
        
#         # 生成唯一标识
#         serial = int(cam.DeviceInfo.GetSerialNumber())
#         cam.SetCameraContext(serial)  # 使用序列号作为上下文标识，传入的参数必须是int型

#     # 创建按时间的保存目录
#     capture_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join(output_root, capture_time)
#     os.makedirs(output_dir)

#     # 开始采集
#     cam_array.StartGrabbing(py.GrabStrategy_LatestImageOnly)
    
#     frame_counts = {int(cam.DeviceInfo.GetSerialNumber()):0 for cam in cam_array}
#     grab_timeout = 5000  # 超时时间5秒

#     while True:
#         # 获取图像结果
#         res = cam_array.RetrieveResult(grab_timeout, py.TimeoutHandling_ThrowException)
        
#         if res.GrabSucceeded():
#             # 获取相机上下文（序列号）
#             serial = res.GetCameraContext()
            
#             # 生成文件名
#             frame_num = frame_counts[serial] + 1
#             filename = f"{serial}_frame{frame_num:04d}.png"
#             save_path = os.path.join(output_dir, filename)
            
#             # 转换并保存图像
#             img = res.Array
#             cv2.imwrite(save_path, img)
            
#             # 更新计数器
#             frame_counts[serial] = frame_num
#             print(f"已保存 {filename}")

#             # 检查采集完成条件（例如每台采集100帧）
#             if all(count >= 10 for count in frame_counts.values()):
#                 break

#         # 释放资源
#         res.Release()

# except py.GenericException as e:
#     print(f"Pylon错误: {e}")
# except KeyboardInterrupt:
#     print("用户中断采集")
# finally:
#     # 安全关闭资源
#     if 'cam_array' in locals():
#         cam_array.StopGrabbing()
#         cam_array.Close()
#     print("采集完成，数据保存在:", output_dir)
