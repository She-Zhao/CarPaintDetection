# host_debug.py

import socket
import cv2
import json
import time
from pathlib import Path
from framework.module.transfer import DataProtocol # 确保路径正确

def run_debug_server():
    HOST = '0.0.0.0'
    PORT = 4097
    # 设定主机的存储根目录
    OUTPUT_ROOT = Path("received_output") 
    OUTPUT_ROOT.mkdir(exist_ok=True)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    
    print(f"🚀 [Host] 接收服务就绪 (Port {PORT})")
    print(f"📂 [Host] 数据将存入: {OUTPUT_ROOT.absolute()}")

    conn, addr = server.accept()
    print(f"🔗 从机已连接: {addr}")

    try:
        while True:
            # 1. 解包数据
            try:
                image, meta_data = DataProtocol.unpack_data(conn)
            except ConnectionError:
                print("⚠️ 连接中断，等待重连...")
                conn.close()
                conn, addr = server.accept() # 简单的重连逻辑
                print(f"🔗 从机重连: {addr}")
                continue
            except Exception as e:
                print(f"❌ 解析错误: {e}")
                break

            # 2. 解析元数据
            # 兼容旧协议：如果 meta_data 是列表（旧版只发了defects list），则手动构造 dummy meta
            if isinstance(meta_data, list):
                print("⚠️ 收到旧版协议数据，跳过")
                continue
            
            pos_id = meta_data.get("pos_id", 0)
            filename = meta_data.get("filename", f"unknown_{time.time()}.png")
            is_last = meta_data.get("is_last", False)
            defects = meta_data.get("defects", [])

            # 3. 创建对应的点位文件夹
            # 类似于 framework/output/pos1
            pos_dir = OUTPUT_ROOT / f"pos{pos_id}"
            pos_dir.mkdir(parents=True, exist_ok=True)

            # 4. 保存图像
            save_path = pos_dir / filename
            cv2.imwrite(str(save_path), image)
            print(f"   [保存] {save_path.name} (Pos {pos_id})")

            # 5. 如果包含缺陷数据（通常在最后一张），保存 defects.json
            if defects:
                json_path = pos_dir / "defects.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(defects, f, indent=2, ensure_ascii=False)
                print(f"   [保存] defects.json (包含 {len(defects)} 个缺陷)")

            # 6. 如果是该组最后一张，打印提示
            if is_last:
                print(f"✅ 点位 {pos_id} 所有数据接收完毕！\n")

    except KeyboardInterrupt:
        print("\n停止服务")
    finally:
        if 'conn' in locals(): conn.close()
        server.close()

if __name__ == "__main__":
    run_debug_server()