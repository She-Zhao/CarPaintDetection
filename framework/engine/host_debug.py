import socket
import cv2
import json
import os
import time
from pathlib import Path
from framework.module.transfer import DataProtocol

def run_debug_server():
    HOST = '0.0.0.0'  # 监听所有网卡
    PORT = 4097       # 必须与从机设置的端口一致
    SAVE_DIR = Path("received_test")
    SAVE_DIR.mkdir(exist_ok=True)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    
    print(f"🚀 [Host] 调试接收服务已启动，监听端口 {PORT}...")
    print(f"📂 [Host] 接收到的文件将保存到 {SAVE_DIR.absolute()}")
    print("⏳ 等待从机连接...")

    conn, addr = server.accept()
    print(f"🔗 从机已连接: {addr}")

    count = 0
    try:
        while True:
            print(f"\n等待数据包 #{count+1} ...")
            
            # 使用协议解包
            try:
                image, json_data = DataProtocol.unpack_data(conn)
            except ConnectionError:
                print("⚠️ 连接中断")
                break
            except Exception as e:
                print(f"❌ 解析错误: {e}")
                break

            # 保存收到的数据进行验证
            timestamp = int(time.time())
            img_path = SAVE_DIR / f"recv_{count}_{timestamp}.png"
            json_path = SAVE_DIR / f"recv_{count}_{timestamp}.json"

            cv2.imwrite(str(img_path), image)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"✅ 收到数据! 图片尺寸: {image.shape}, 缺陷数量: {len(json_data)}")
            print(f"   已保存: {img_path.name}")
            count += 1

    except KeyboardInterrupt:
        print("停止服务")
    finally:
        conn.close()
        server.close()

if __name__ == "__main__":
    run_debug_server()