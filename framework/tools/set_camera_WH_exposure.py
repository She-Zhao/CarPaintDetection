# -*-coding:utf-8 -*-
from pypylon import pylon

def search_get_device(camera_ip):#查找相机

    tl_factory = pylon.TlFactory.GetInstance()
    for dev_info in tl_factory.EnumerateDevices():
        if dev_info.GetIpAddress() == str(camera_ip):
            print("DeviceClass:", dev_info.GetDeviceClass())
            print(f"ModelName:{dev_info.GetModelName()}\n"f"IP:{dev_info.GetIpAddress()}")
            camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
            break
    else:
        raise EnvironmentError("no GigE device found")
    return camera

# Height = Width = 800
Height = 2048
Width = 2448
ExposureTime = 8000

cam1 = search_get_device('10.18.18.25')
cam2 = search_get_device('10.18.18.26')
cam1.Open()
cam2.Open()

cam1.Height.Value = Height
cam1.Width.Value = Width
cam2.Height.Value = Height
cam2.Width.Value = Width

cam1.ExposureTime.SetValue(ExposureTime)
cam2.ExposureTime.SetValue(ExposureTime)

cam1.Close()
cam2.Close()
