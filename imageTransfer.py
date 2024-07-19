from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

# 将tif 文件转化为 png 文件
def saveTiF2Png(filedir,savedir):
    img = cv2.imread(filedir)
    cv2.imshow(filedir,img)
    resized_img = cv2.resize(img, (512,512)) # 使用cv2.resize()调整图像尺寸
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) # 图片转化为灰度
    cv2.imwrite(savedir, gray_img) # 保存调整尺寸后的图像
    
#将单通道('I') 的图片 转化为单通道('L')的图片
"""
为什么要转换：
train_data/imgs_png_c1/frame_0.png
通道数量: ('L',)
亮度值低于50的像素点数量占比 0.9933128356933594
train_data/masks/frame_0.tif
通道数量: ('I',)
亮度值低于50的像素点数量占比 1.0
test_data/imgs_png/frame_0.png
通道数量: ('L',)
亮度值低于50的像素点数量占比 0.9934539794921875

"""
def convertI2L(filedir,savedir):
    # 读取单通道'I'模式图像
    img = Image.open(filedir).convert('I')
    # 将'I'模式图像转换为'L'模式图像
    gray_img = img.convert('L')
    # 保存转换后的'L'模式图像
    gray_img.save(savedir)

# 指定文件夹路径
folder_path = 'train_data/masks_png_c4'
save_path = 'train_data/masks_png_c1'

# 初始化一个空列表
train_masks_tif_files = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件是否以 .tif 结尾
    if file_name.endswith('.png'):
        # 将文件名添加到列表中
        train_masks_tif_files.append(file_name)
        saveName = file_name.split(".")[0]+".png"
        
        print("file dir = ",folder_path+"/"+file_name)
        print("save dir = ", save_path + "/" + saveName)
        saveTiF2Png(folder_path+"/"+file_name, save_path + "/" + saveName)
        # convertI2L(folder_path+"/"+file_name, save_path + "/" + saveName)
