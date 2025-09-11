# import os
# print(os.path.getsize("data\CVOGL_DroneAerial\CVOGL_DroneAerial_train.pth"))
#
# import torch
# data = torch.load('data/CVOGL_DroneAerial/CVOGL_DroneAerial_train.pth')
# for i, item in enumerate(data[:5]):
#     print(f"样本{i}: {item}")
import torch
data = torch.load('data/CVOGL_DroneAerial/CVOGL_DroneAerial_val.pth')
print(data[0])