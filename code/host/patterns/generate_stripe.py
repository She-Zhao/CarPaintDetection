import numpy as np
import cv2

def generate_stripes_row(width, height, num_stripes):
    image = np.zeros((height, width), dtype=np.uint8)
    stripe_width = width // num_stripes
    for i in range(num_stripes):
        if i % 2 == 0:
            color = 255  # white
        else:
            color = 0    # black
        x0 = i * stripe_width
        x1 = x0 + stripe_width
        image[:, x0:x1] = color
        if i == num_stripes-1:
            if color == 0:
                image[:, x1:] = 255 
            else:
                image[:, x1:] = 0
    return image


def generate_stripes_col(width, height, num_stripes):
    image = np.zeros((height, width), dtype=np.uint8)
    stripe_height = height // num_stripes
    for i in range(num_stripes):
        if i % 2 == 0:
            color = 255  # white
        else:
            color = 0    # black
        y0 = i * stripe_height
        y1 = y0 + stripe_height
        image[y0:y1, :] = color
        if i == num_stripes-1:
            if color == 0:
                image[y1:, :] = 255 
            else:
                image[y1:, :] = 0        

    return image

def roll_image_right(image, num_pixels):
    # 获取图像的宽度和高度
    height, width = image.shape[:2]
    # 向右滚动平移 num 个像素
    rolled_image = np.roll(image, num_pixels, axis=1)
    # 将最右边的 num 列移到最左边
    rolled_image[:, :num_pixels] = image[:, width - num_pixels:]
    return rolled_image


def roll_image_down(image, num_pixels):
    # 向下滚动平移 num 个像素
    rolled_image = np.roll(image, num_pixels, axis=0)
    # 获取图像的宽度和高度
    height, width = image.shape[:2]
    # 将最下面的 num 行移到最上面
    rolled_image[:num_pixels, :] = image[height - num_pixels:, :]
    return rolled_image


num_stripes_row = 100                           #横向条纹周期数
num_stripes_col = 60                            #纵向条纹周期数

stripe_width_row = 2160 // num_stripes_col      #横向条纹的周期
stripe_width_col = 3840 // num_stripes_row      #纵向条纹的周期


# 横向条纹
image_row = generate_stripes_row(3840, 2160, num_stripes_row)   
image_row_1 = roll_image_right(image_row.copy(), int(stripe_width_row*0.5))
image_row_2 = roll_image_right(image_row.copy(), stripe_width_row*1)    
image_row_3 = roll_image_right(image_row.copy(), int(stripe_width_row*1.5))


# 纵向条纹
image_col = generate_stripes_col(3840, 2160, num_stripes_col)
image_col_1 = roll_image_down(image_col.copy(), int(stripe_width_col*0.5))
image_col_2 = roll_image_down(image_col.copy(), stripe_width_col*1)
image_col_3 = roll_image_down(image_col.copy(), int(stripe_width_col*1.5))



cv2.imwrite(f"row_{num_stripes_row}_0.png", image_row)
cv2.imwrite(f"row_{num_stripes_row}_1.png", image_row_1)
cv2.imwrite(f"row_{num_stripes_row}_2.png", image_row_2)
cv2.imwrite(f"row_{num_stripes_row}_3.png", image_row_3)


cv2.imwrite(f"col_{num_stripes_col}_0.png", image_col)
cv2.imwrite(f"col_{num_stripes_col}_1.png", image_col_1)
cv2.imwrite(f"col_{num_stripes_col}_2.png", image_col_2)
cv2.imwrite(f"col_{num_stripes_col}_3.png", image_col_3)