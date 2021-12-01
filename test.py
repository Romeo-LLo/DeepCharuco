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
import cv2
import random
from Model import DeepCharuco
import matplotlib

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.csv = pd.read_csv('output.csv')
        self.sample_num = self.csv.shape[0]       # 以csv的數量為主，照片可能會比較多
        files = glob.glob(os.path.join(root, '*.jpg'))
        files.sort()
        self.files = files[:self.sample_num]
        self.len = len(self.files)

        img_fn = self.files[0]
        img = Image.open(img_fn)
        self.cell_size = 8
        self.width = img.size[0]
        self.height = img.size[1]
        self.x_cells = int(self.width / self.cell_size)
        self.y_cells = int(self.height  / self.cell_size)

    def __getitem__(self, index):
        img_fn = self.files[index]
        img = Image.open(img_fn)
        coords = self.csv.iloc[index, 1:]
        label2D = self.coord2binary(coords)
        id2D = self.idto2D(coords)

        if self.transform is not None:
            img = self.transform(img)

        return img, label2D, id2D

    def coord2binary(self, coords):
        label2D = torch.zeros(self.height, self.width)  # 480*640
        for i in range(4):
            y = round(coords[2*i+1])
            x = round(coords[2*i])
            label2D[y, x] = 1
        return label2D

    def idto2D(self, coords):
        id2D = torch.zeros(self.y_cells, self.x_cells)  # 0 stands for no id
        for i in range(4):
            x = round(coords[2*i] // self.cell_size)
            y = round(coords[2*i+1] // self.cell_size)
            id2D[y, x] = i + 1
        return id2D


    def __len__(self):
        return self.len

def idto2D(coords):
    cell_size = 8
    y_cells, x_cells = 60, 80
    id2D = torch.zeros(y_cells, x_cells)  # 0 stands for no id
    for i in range(4):
        x = round(coords[2*i] // cell_size)
        y = round(coords[2*i+1] // cell_size)
        id2D[y, x] = i + 1
    return id2D

def imshow(img):
    img = img.numpy()
    print(img.shape)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def imgValid(img, id2D):
    id = id2D[0]
    showimg = img.numpy()
    for h in range(60):
        for w in range(80):
            index = id[h, w]   # 0 is the first item of the batch

            if index != 0:
                y = h * 8 + 4
                x = w * 8 + 4
                print(index, y, x)
                plt.text(x, y, str(int(index.item())), fontsize=5, bbox=dict(facecolor="r"))

    plt.imshow(np.transpose(showimg.squeeze(0), (1, 2, 0)), cmap='gray')
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

def SetupTrain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)
    return model




def test_show(model_dir):
    testset = CustomDataset(root='TestImage', transform=transforms.ToTensor())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
    checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    # cap = cv2.VideoCapture(0)
    #
    # while (True):
    #     ret, frame = cap.read()
    #
    #     imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     transform = transforms.ToTensor()
    #     imgGray = transform(imgGray).unsqueeze(0).to(device)
    #
    #     out_loc = model(imgGray)['semi']
    #     out_id = model(imgGray)['desc']
    #
    #     pred_loc = torch.argmax(out_loc, dim=1)
    #     pred_id = torch.max(out_id, dim=1)[1].cpu().numpy()
    #     for h in range(60):
    #         for w in range(80):
    #
    #             y = h * 8
    #             x = w * 8
    #             index = pred_id[0, h, w]
    #             if index != 0:
    #                 frame = cv2.putText(frame, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
    #                                       cv2.LINE_AA)
    #
    #     cv2.imshow('id img', frame)

    csv = pd.read_csv('test_output.csv')

    for i in range(10):
        id = random.randint(1, 7000)
        filename = 'TrainImage/{:04d}.jpg'.format(id)
        coords = csv.iloc[i, 1:]
        target_id = idto2D(coords).unsqueeze(0).to(device)

        # filename = 'TestImage/{:04d}.jpg'.format(id)
        img = Image.open(filename)

        transform = transforms.ToTensor()
        img = transform(img).unsqueeze(0).to(device)
        # img = transform(img).to(device)

        out_loc = model(img)['semi']
        out_id = model(img)['desc']
        criterion = nn.CrossEntropyLoss()
        id_loss = criterion(out_id, target_id.type(torch.int64))

        print("id_loss = {:.4f}".format(id_loss))
        pred_loc = torch.argmax(out_loc, dim=1)
        pred_id = torch.max(out_id, dim=1)[1].cpu().numpy()
        showimg = Image.open(filename)
        showimg = transform(showimg).unsqueeze(0)
        showimg = showimg.numpy()

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(np.transpose(showimg.squeeze(0), (1, 2, 0)), cmap='gray')
        cms = matplotlib.cm

        # show all discovered
        c = 0
        for h in range(60):
            for w in range(80):
                x = w * 8
                y = h * 8
                loc = pred_loc[0, h, w].item()
                if loc != 64:
                    x_ = loc % 8 + x
                    y_ = loc // 8 + y
                    circ = plt.Circle((x_, y_), 3, fill=True, color=cms.jet(0.9))
                    ax.add_patch(circ)

                index = pred_id[0, h, w]
                if index != 0:
                    c += 1
                    plt.text(x, y, str(int(index.item())), fontsize=15, color="yellow")
        # id_list = [1, 2, 3, 4]
        # for h in range(60):
        #     for w in range(80):
        #         x = w * 8
        #         y = h * 8
        #         index = pred_id[0, h, w]
        #         if index in id_list:
        #             id_list.remove(index)
        #             plt.text(x, y, str(int(index.item())), fontsize=15, color="yellow")
        #
        #         loc = pred_loc[0, h, w].item()
        #         if loc != 64:
        #             x_ = loc % 8 + x
        #             y_ = loc // 8 + y
        #             circ = plt.Circle((x_, y_), 3, fill=True, color=cms.jet(0.9))
        #             ax.add_patch(circ)

        plt.show()

def test_loc_loss():
    testset = CustomDataset(root='TestImage', transform=transforms.ToTensor())
    testset_loader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCharuco()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
    checkpoint = torch.load('Model_dict/epoch40.pth', map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()


    for batch_id, (input, target_label2D, target_id) in enumerate(testset_loader):
        input, target_label2D, target_id = input.to(device), target_label2D.to(device), target_id.to(device)
        optimizer.zero_grad()
        pred_loc = model(input)['semi']
        pred_id = model(input)['desc']

        target_loc = labels2Dto3D_flattened(target_label2D.unsqueeze(1), 8)

if __name__ == "__main__":
    model_dir = 'Model_dict/epoch100.pth'
    test_show(model_dir)

