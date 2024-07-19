from PIL import Image
import matplotlib.pyplot as plt
import os

# 将tif 文件转化为 png 文件
def saveTiF2Png(filedir,savedir):
    img = Image.open(filedir).resize((512, 512)).convert('L')
    # 使用 matplotlib 保存图像
    plt.figure(figsize=(6, 6))
    plt.imsave(savedir,img,cmap='gray')

# 指定文件夹路径
folder_path = 'test_data/predict_masks'
save_path = 'test_data/predict_result2'

# 初始化一个空列表
train_masks_tif_files = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件是否以 .ttf 结尾
    if file_name.endswith('.png'):
        # 将文件名添加到列表中
        train_masks_tif_files.append(file_name)
        saveName = file_name.split(".")[0]+".png"
        
        print("file dir = ",folder_path+"/"+file_name)
        print("save dir = ", save_path + "/" + saveName)
        saveTiF2Png(folder_path+"/"+file_name, save_path + "/" + saveName)
