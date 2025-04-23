import os
import shutil

def delete_output_folders(root_directory):
    # 获取目录下所有子目录
    subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    for subdirectory in subdirectories:
        output_folder_path = os.path.join(root_directory, subdirectory, "output")

        # 检查 output 文件夹是否存在，如果存在则删除
        if os.path.exists(output_folder_path):
            shutil.rmtree(output_folder_path)
            print(f"Deleted {output_folder_path}")

if __name__ == "__main__":
    directory_path = r"D:\Project\PMD\PMD2.4\img_200"
    delete_output_folders(directory_path)
    print("Script execution completed.")
