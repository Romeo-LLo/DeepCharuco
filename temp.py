import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import glob
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torchvision.models as models
import pandas as pd

# #
# img = Image.open('TrainImage/0001.jpg')
# imgGray = ImageOps.grayscale(img)
# print(imgGray.size)
# #
# csv = pd.read_csv('output.csv')
# coords = csv.iloc[1, 1:]
# print(type(coords))
# label2D = torch.zeros([640, 480])
# label2D[3, 6] = 1
# label2D[2, 4] = 1

x_input = torch.randn(16, 60, 64)
y_input = torch.randn(16, 60, 64)
z = torch.stack((x_input, y_input), 1)
y_target = torch.tensor([[[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]]])
crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(x_input,y_target)
print('crossentropyloss_output:\n',crossentropyloss_output)

# for i in range(4):
#     x = round(coords[2 * i])
#     y = round(coords[2 * i + 1])
#     label2D[x, y] = 1
# print((label2D != 0).nonzero(as_tuple=True))
# label2D = torch.zeros([3, 2, 5])
# label2D[:, :, 0] = 1
# print(label2D)

