import requests

'''
上传图片
'''
def upload_image(image_path, datas,upload_url):
    try:
        with open(image_path, 'rb') as file:
            files = {'image': file}
            response = requests.post(upload_url, files=files,data=datas)
            if response.status_code == 200:
                print(f"图片上传成功 => {response.text}")
            else:
                print(f"上传图片失败 , 状态码 => {response.status_code}")

    except Exception as e:
        print(f"异常 => {e}")