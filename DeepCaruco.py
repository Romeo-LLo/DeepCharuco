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

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.csv = pd.read_csv('output.csv')
        self.sample_num = self.csv.shape[0]
        files = glob.glob(os.path.join(root, '*.jpg'))
        files.sort()
        self.files = files[:self.sample_num]
        self.len = len(self.files)
    def __getitem__(self, index):
        img_fn = self.files[index]
        img = Image.open(img_fn)
        coords = self.csv.iloc[index, 1:]

        if self.transform is not None:
            img = self.transform(img)
            coords = torch.tensor(coords)
        return img, coords
    def __len__(self):
        return self.len
def imshow(img):
    img = img.numpy()
    print(img.shape)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


trainset = CustomDataset(root='TrainImage/', transform=transforms.ToTensor())
trainset_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1)
imgs, coords = iter(trainset_loader).next()
imshow(torchvision.utils.make_grid(imgs, nrow=4))

class DeepCharuco(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256

        det_h = 65
        gn = 64
        useGn = False
        self.reBn = True
        if self.reBn:
            print("model structure: relu - bn - conv")
        else:
            print("model structure: bn - relu - conv")

        if useGn:
            print("apply group norm!")
        else:
            print("apply batch norm!")

        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2d(c1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2d(c2)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2d(c3)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2d(c4)

        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.GroupNorm(det_h, det_h) if useGn else nn.BatchNorm2d(65)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.GroupNorm(gn, d1) if useGn else nn.BatchNorm2d(d1)
    def forward(self, x, subpixel=False):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """

        # Let's stick to this version: first BN, then relu
        if self.reBn:
            # Shared Encoder.
            x = self.relu(self.bn1a(self.conv1a(x)))
            conv1 = self.relu(self.bn1b(self.conv1b(x)))
            x, ind1 = self.pool(conv1)
            x = self.relu(self.bn2a(self.conv2a(x)))
            conv2 = self.relu(self.bn2b(self.conv2b(x)))
            x, ind2 = self.pool(conv2)
            x = self.relu(self.bn3a(self.conv3a(x)))
            conv3 = self.relu(self.bn3b(self.conv3b(x)))
            x, ind3 = self.pool(conv3)
            x = self.relu(self.bn4a(self.conv4a(x)))
            x = self.relu(self.bn4b(self.conv4b(x)))
            # Detector Head.
            cPa = self.relu(self.bnPa(self.convPa(x)))
            semi = self.bnPb(self.convPb(cPa))
            # Descriptor Head.
            cDa = self.relu(self.bnDa(self.convDa(x)))
            desc = self.bnDb(self.convDb(cDa))
        else:
            # Shared Encoder.
            x = self.bn1a(self.relu(self.conv1a(x)))
            x = self.bn1b(self.relu(self.conv1b(x)))
            x = self.pool(x)
            x = self.bn2a(self.relu(self.conv2a(x)))
            x = self.bn2b(self.relu(self.conv2b(x)))
            x = self.pool(x)
            x = self.bn3a(self.relu(self.conv3a(x)))
            x = self.bn3b(self.relu(self.conv3b(x)))
            x = self.pool(x)
            x = self.bn4a(self.relu(self.conv4a(x)))
            x = self.bn4b(self.relu(self.conv4b(x)))
            # Detector Head.
            cPa = self.bnPa(self.relu(self.convPa(x)))
            semi = self.bnPb(self.convPb(cPa))
            # Descriptor Head.
            cDa = self.bnDa(self.relu(self.convDa(x)))
            desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        output = {'semi': semi, 'desc': desc}

        return output

def SetupTrain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)

def train(model):
    running_losses = []
    epoch = 0

    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    lambda2 = lambda epoch: 0.98 ** epoch
    scheduler = LambdaLR(optimizer, lr_lambda=lambda2)
    loc_criterion = nn.CrossEntropyLoss()
    id_criterion = nn.CrossEntropyLoss()
    model.train()

    interval = 100
    iteration = 0
    loss = 0
    for ep in range(epoch):
        for batch_id, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            _, pred = torch.max(output, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if iteration % interval == 0:
                print("Train epoch {}  [{}/{}] {:.0f}%]\tLoss: {:.6f}".format(
                    ep, batch_id*len(data), len(trainset_loader.dataset),
                    100 * batch_id / len(trainset_loader), loss.item()))
            iteration += 1
        test(model, testset_loader, device)
        scheduler.step()
    save_checkpoint('Model_dict/resnet152.pth', model, optimizer)