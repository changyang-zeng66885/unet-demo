import torch
# 检查是否支持CUDA
if torch.cuda.is_available():
    print(torch.version.cuda)
else:
    print("CUDA 不可用")