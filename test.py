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
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import pandas as pd
import tqdm


# img = Image.open('Train')
csv = pd.read_csv('output.csv')
coords = csv.iloc[0, 1:]
print(coords.shape)
label2D = torch.zeros([640, 480])

for i in range(4):
    x = round(coords[2 * i])
    y = round(coords[2 * i + 1])
    label2D[x, y] = 1
print ((label2D == 1).nonzero(as_tuple=True)[1])