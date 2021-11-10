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
import tqdm
import cv2

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
        imgGray = ImageOps.grayscale(img)
        coords = self.csv.iloc[index, 1:]
        label2D = self.coord2binary(coords, imgGray.size)
        id2D = self.idto2D(coords, imgGray.size)



        if self.transform is not None:
            imgGray = self.transform(imgGray)
            # label2D =self.transform(label2D)  # already torch
            # id2D =self.transform(id2D)

        return imgGray, label2D, id2D

    def coord2binary(self, coords, img_size):
        label2D = torch.zeros([img_size[1], img_size[0]])
        for i in range(4):
            x = round(coords[2*i])
            y = round(coords[2*i+1])
            label2D[x, y] = 1
        return label2D

    def idto2D(self, coords, img_size):
        id2D = torch.zeros([img_size[1]//8, img_size[0]//8])  # 0 stands for no id
        for i in range(4):
            x = round(coords[2*i]) // 8
            y = round(coords[2*i+1]) // 8
            id2D[x, y] = i+1
        return id2D


    def __len__(self):
        return self.len
def imshow(img):
    img = img.numpy()
    print(img.shape)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def labels2Dto3D_flattened(labels, cell_size):

    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    labels = space2depth(labels).cuda()
    dustbin = torch.ones((batch_size, 1, Hc, Wc)).cuda()
    # labels = torch.cat((labels*2, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)  # why times 2
    labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)

    labels = torch.argmax(labels, dim=1)
    return labels

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output



class DeepCharuco(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 5

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
    return model

def save_checkpoint(checkpoint_path, model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print("Model saved")



def train():
    running_losses = []
    epoch = 40
    beta = 0.8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    lambda2 = lambda epoch: 0.98 ** epoch
    scheduler = LambdaLR(optimizer, lr_lambda=lambda2)
    criterion = nn.CrossEntropyLoss()
    model.train()

    interval = 5
    loc_loss = 0
    id_loss = 0
    for ep in range(epoch):
        iteration = 0
        for batch_id, (input, target_label2D, target_id) in enumerate(trainset_loader):
            input, target_label2D, target_id = input.to(device), target_label2D.to(device), target_id.to(device)
            optimizer.zero_grad()
            pred_loc = model(input)['semi']
            pred_id = model(input)['desc']
            print(pred_loc.shape)
            print(pred_id.shape)
            target_loc = labels2Dto3D_flattened(target_label2D.unsqueeze(1), 8)
            loc_loss = criterion(pred_loc, target_loc.type(torch.int64))
            id_loss = criterion(pred_id, target_id.type(torch.int64))

            loss = loc_loss + beta * id_loss
            loss.backward()
            optimizer.step()
            if iteration % interval == 0:
                print("Train epoch {}  [{}/{}] {:.0f}%]\tLoss: {:.6f}".format(
                    ep, batch_id*len(target_label2D), len(trainset_loader.dataset),
                    100 * batch_id / len(trainset_loader), loss.item()))
            iteration += 1
        scheduler.step()
    # save_checkpoint('Model_dict/1st_version.pth', model, optimizer)

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    checkpoint = torch.load('Model_dict/1st_version.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()


    # cap = cv2.VideoCapture(0)
    #
    # while (True):
    #     ret, frame = cap.read()
    #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).to(device)
    #     out_loc = model(frame_gray)['semi']
    #     out_id = model(frame_gray)['desc']
    #
    #     pred_loc = torch.max(out_loc, dim=1)

    showimg = cv2.imread('TrainImage/2.jpg')

    img = Image.open('TrainImage/1.jpg')
    imgGray = ImageOps.grayscale(img)
    transform = transforms.ToTensor()
    imgGray = transform(imgGray).unsqueeze(0).to(device)

    out_loc = model(imgGray)['semi']
    out_id = model(imgGray)['desc']

    pred_loc = torch.argmax(out_loc, dim=1)
    pred_id = torch.max(out_id, dim=1)[1].cpu().numpy()
    for h in range(60):
        for w in range(80):
            y = h * 8
            x = w * 8
            index = pred_id[0, h, w]
            showimg = cv2.putText(showimg, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('id img', showimg)


    cv2.waitKey(0)
    cv2.destroyAllWindows()






if __name__ == "__main__":
    trainset = CustomDataset(root='TrainImage/', transform=transforms.ToTensor())
    trainset_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1)
    imgs, coords, id = iter(trainset_loader).next()
    # # imshow(torchvision.utils.make_grid(imgs, nrow=4))
    # train()
    test()
