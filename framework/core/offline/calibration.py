import cv2
import numpy as np
import glob
import os
from stitched2 import stitch_images, calculate_homography

# ---------------------- 参数设置 ----------------------
chessboard_size = (9, 6)  # 棋盘格内角点数量（宽x高）
square_size = 0.014  # 棋盘格方格边长（单位：米）
image_dir_left = "../../data/factory/12/2/*.bmp" #左相机图片路径
image_dir_right = "../../data/factory/12/1/*.bmp" #右相机图片路径
params_save_folder = "../../data/factory" #内外参保存文件夹

# ---------------------- 内参保存函数 ----------------------
def save_intri_params(k, dist, image_dir):
    dir_path = os.path.dirname(image_dir)
    last_folder = os.path.basename(dir_path)+'_intri.txt'
    intri_path = os.path.join(params_save_folder, last_folder)
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]
    intri = [fx, fy, cx, cy, k1, k2, k3, p1, p2]
    with open(intri_path, "w", encoding="utf-8") as f:
        for item in intri:
            f.write(f"{item}\n")  # 换行写入

# ---------------------- 外参保存函数 ----------------------
def save_extri_params(file_path, RT):
    np.savetxt(file_path, RT, fmt="%.4f", delimiter=" ")

# ---------------------- 重投影误差计算函数 ----------------------
def compute_reprojection_error(obj_points, img_points, rvecs, tvecs, K, dist):
    total_error = 0
    total_points = 0

    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(obj_points[i])

    mean_error = np.sqrt(total_error / total_points)
    return mean_error

# ---------------------- 单目标定函数 ----------------------
def calibrate_camera(image_paths, chessboard_size, square_size):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    obj_points = []
    img_points = []

    for path in image_paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            obj_points.append(objp)
            img_points.append(corners_refined)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    mean_error = compute_reprojection_error(obj_points, img_points, rvecs, tvecs, K, dist)

    return ret, K, dist, rvecs, tvecs, obj_points, img_points, mean_error


# ---------------------- 左相机标定 ----------------------
left_images = sorted(glob.glob(image_dir_left))
ret_left, K_left, dist_left, rvecs_left, tvecs_left, obj_points_left, img_points_left, error_left = calibrate_camera(left_images, chessboard_size, square_size)
print("\n左相机内参:\n", K_left)
print("左相机畸变系数:\n", dist_left)
print(f"左相机重投影误差: {error_left:.4f} px\n")
save_intri_params(K_left, dist_left.flatten(), image_dir_left)

# ---------------------- 右相机标定 ----------------------
right_images = sorted(glob.glob(image_dir_right))
ret_right, K_right, dist_right, rvecs_right, tvecs_right, obj_points_right, img_points_right, error_right = calibrate_camera(right_images, chessboard_size, square_size)
print("右相机内参:\n", K_right)
print("右相机畸变系数:\n", dist_right)
print(f"右相机重投影误差: {error_right:.4f} px\n")
save_intri_params(K_right, dist_right.flatten(), image_dir_right)

# ---------------------- 双目标定准备数据 ----------------------
obj_points = []
img_points_left_stereo = []
img_points_right_stereo = []

for i in range(len(left_images)):
    img_left = cv2.imread(left_images[i])
    img_right = cv2.imread(right_images[i])

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        corners_left_refined = cv2.cornerSubPix(
            gray_left, corners_left, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        corners_right_refined = cv2.cornerSubPix(
            gray_right, corners_right, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        obj_points.append(objp)
        img_points_left_stereo.append(corners_left_refined)
        img_points_right_stereo.append(corners_right_refined)

# ---------------------- 双目标定 ----------------------
ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left_stereo, img_points_right_stereo,
    K_left, dist_left, K_right, dist_right,
    gray_left.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("双目标定旋转矩阵 R:\n", R)
print("双目标定平移向量 T:\n", T)
RT = np.hstack((R, T))
save_name_for_ext = os.path.basename(os.path.dirname(image_dir_left))+os.path.basename(os.path.dirname(image_dir_right))
save_extri_params(os.path.join(params_save_folder, save_name_for_ext)+ "_extri.txt", RT)

# ---------------------- 双目重投影误差 ----------------------
def compute_stereo_reprojection_error(obj_points, img_points_left, img_points_right,
                                      rvecs_left, tvecs_left, K_left, dist_left,
                                      rvecs_right, tvecs_right, K_right, dist_right):
    total_error = 0
    total_points = 0

    for i in range(len(obj_points)):
        img_points_proj_left, _ = cv2.projectPoints(obj_points[i], rvecs_left[i], tvecs_left[i], K_left, dist_left)
        error_left = cv2.norm(img_points_left[i], img_points_proj_left, cv2.NORM_L2)

        img_points_proj_right, _ = cv2.projectPoints(obj_points[i], rvecs_right[i], tvecs_right[i], K_right, dist_right)
        error_right = cv2.norm(img_points_right[i], img_points_proj_right, cv2.NORM_L2)

        total_error += (error_left ** 2 + error_right ** 2)
        total_points += 2 * len(obj_points[i])

    mean_error = np.sqrt(total_error / total_points)
    return mean_error

stereo_error = compute_stereo_reprojection_error(
    obj_points, img_points_left_stereo, img_points_right_stereo,
    rvecs_left, tvecs_left, K_left, dist_left,
    rvecs_right, tvecs_right, K_right, dist_right
)

print(f"\n双目标定重投影误差: {stereo_error:.4f} px")

# ---------------------- 到这里标定+误差计算完整； 开始拼接 ----------------------

# ---------------------- 计算单应性矩阵 H (右到左) ----------------------
points_left = np.vstack(img_points_left_stereo).reshape(-1, 2)
points_right = np.vstack(img_points_right_stereo).reshape(-1, 2)

H, mask = cv2.findHomography(points_right, points_left, cv2.RANSAC, 5.0)

print("\n右相机到左相机的单应性矩阵 H:\n", H)
save_extri_params(os.path.join(params_save_folder, save_name_for_ext)+ "_H.txt",H)




# ---------------------- 验证H映射效果（可选） ----------------------
sample_img_right = cv2.imread(right_images[0])
sample_img_left = cv2.imread(left_images[0])


stitched_img = stitch_images(sample_img_left, sample_img_right, H)
# 显示拼接结果
cv2.imshow("Stitched Image", stitched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
