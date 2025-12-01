import cv2
import numpy as np

def stitch_images(img_left, img_right, H):
    """
    使用单应性矩阵将左右图像拼接在一起

    参数:
    img_left (ndarray): 左相机图像
    img_right (ndarray): 右相机图像
    H (ndarray): 右图到左图的单应性矩阵

    返回:
    stitched_img (ndarray): 拼接后的图像
    """
    # 获取图像尺寸
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    # 右图四个角坐标
    corners_right = np.array([
        [0, 0],
        [w_right, 0],
        [w_right, h_right],
        [0, h_right]
    ], dtype='float32').reshape(-1, 1, 2)

    # 单应变换后的坐标
    warped_corners_right = cv2.perspectiveTransform(corners_right, H)

    # 左图四个角坐标
    corners_left = np.array([
        [0, 0],
        [w_left, 0],
        [w_left, h_left],
        [0, h_left]
    ], dtype='float32').reshape(-1, 1, 2)

    # 拼接后整体四个角
    all_corners = np.concatenate((warped_corners_right, corners_left), axis=0)

    # 求出 canvas 尺寸
    [x_min, y_min] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    [x_max, y_max] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    # canvas 大小
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    # 偏移矩阵 T
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    # 变换右图
    warped_right = cv2.warpPerspective(img_right, T @ H, (canvas_width, canvas_height))

    # 创建空白拼接图
    stitched_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 将左图贴上去
    stitched_img[-y_min:h_left - y_min, -x_min:w_left - x_min] = img_left

    # 把右图覆盖进 stitched_img 中（非零像素位置覆盖）
    mask = (warped_right > 0)
    stitched_img[mask] = warped_right[mask]

    return stitched_img

if __name__ == "__main__":
    # 读入左右相机图片
    img_left = cv2.imread("../../data/factory/123/2.bmp")
    img_right = cv2.imread("../../data/factory/123/1.bmp")

    # 单应性矩阵
    H = np.array([
        [1.06691913e+00, 1.01931121e-01, -7.46439498e+02],
        [5.61735518e-02, 1.07194953e+00, -5.49832197e+00],
        [5.42599169e-06, 4.62350815e-05, 1.00000000e+00]
    ])
    # 调用拼接函数
    stitched_img = stitch_images(img_left, img_right, H)

    # 显示拼接结果
    cv2.imshow("Stitched Image", stitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


