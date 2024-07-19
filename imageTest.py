from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# # 统计亮度值低于50的像素点数量
def calculateLowBrightnessRate(imagePath):
    img = Image.open(imagePath)
    # 将图像转换为numpy数组
    img_array = np.array(img)

    low_brightness_count = np.sum(img_array < 50)

    # 计算占比
    total_pixels = img_array.size
    low_brightness_ratio = low_brightness_count / total_pixels
    return low_brightness_ratio

initPath = 'train_data/masks/frame_0.tif'
rate1 = calculateLowBrightnessRate(initPath)
print(f"低亮度像素点占比: {rate1:.2%}") # 低亮度像素点占比: 100.00%
    

# 将原图像通过PIL和pyplot保存
imgInit = Image.open(initPath)
plt.figure(figsize=(8, 6))
plt.imshow(imgInit , cmap='gray')
plt.axis('off')
plt.savefig('saved_image_by_plt.png', dpi=300, bbox_inches='tight') # 保存的图像是正常的
imgInit.save('saved_image_by_pil.png') #保存的图像时全黑的

##再次测试用打开plt保存的图像
imgP = Image.open('saved_image_by_pil.png')
rateByPyplot = calculateLowBrightnessRate('saved_image_by_plt.png')
print(f"Pyplot 低亮度像素点占比: {rateByPyplot:.2%}") ## Pyplot 低亮度像素点占比: 67.92%

# 上面的实验说明，可能需要将原来的mask通过pyplot转化一下，这样才能看到比较清晰的图片。
#　后面可用试一试这种方法，将train_data/test_data 中的masks数据进行数据预处理。



