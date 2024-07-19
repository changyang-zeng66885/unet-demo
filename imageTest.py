from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 展示图片
# def showImage(imagePath):
#     img = Image.open(imagePath)
#     plt.figure(figsize=(8, 6))
#     plt.imshow(img , cmap='gray')
#     plt.axis('off')
#     plt.show() 

# # 统计亮度值低于50的像素点占比
def calculateLowBrightnessRate(imagePath):
    img = cv2.imread(imagePath)

    # 转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray_img",gray_img)
    # 将图像转换为numpy数组
    img_array = np.array(gray_img)
    low_brightness_count = np.sum(img_array < 50)
    # 计算占比
    total_pixels = img_array.size
    low_brightness_ratio = low_brightness_count / total_pixels
    return low_brightness_ratio

#　打印图片的通道数量
def getChannleNum(imagePath):
    img = Image.open(imagePath)
    num_channels = img.getbands()
    return num_channels
    

    
filePathList = [
    "train_data/imgs_png_c1/frame_0.png",
    "train_data/masks_png_c1/frame_0_mask.png",
    "test_data/imgs_png/frame_0.png"
]

for imagePath in filePathList:
    print(imagePath)
    print("通道数量:",getChannleNum(imagePath))
    print("亮度值低于50的像素点数量占比",calculateLowBrightnessRate(imagePath))
    img = cv2.imread(imagePath)
    height, width,_ = img.shape
    print(f"图片尺寸: 宽 {width} px , 高 {height} px")
    """
    cv2.imread(imagePath) 函数读取的图像默认是 RGB 三通道格式。
    即使输入的图像是单通道灰度图,它也会被转换成 RGB 三通道格式。
    因此,img.shape 返回的通道数是 3。
    """
    # cv2.imshow(imagePath,img)
    # showImage(imagePath)

    



