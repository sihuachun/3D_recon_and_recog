import dxchange
import tomopy
import numpy as np
import glob
import warnings

path = './NUS 3D 数据/'
shape = (80, 728, 728)
warnings.filterwarnings("ignore")
list_similar_path = []
dict_wrong_path = {}


def recon(path, similar=False):
    """
    reconstruction object
    :param similar: boolean, specify to or not to create reconstructed objedct
    :param path: image location
    :return: 3-D object
    """

    obj = dxchange.reader.read_tiff(fname=path, slc=None)  # Generate an object

    ang = tomopy.angles(180)  # Generate uniformly spaced tilt angles

    # test
    obj = obj.astype(np.float32)
    # print(obj.shape)
    if similar:
        obj += np.random.uniform(size=obj.shape, low=-0.2, high=0.2)

    sim = tomopy.project(obj, ang)  # Calculate projections
    rec = tomopy.recon(sim, ang, algorithm='art')  # Reconstruct object
    # print(rec.shape)

    return rec


def create_similar_couple(file_path):
    """
    相似样本对
    :param file_path: './NUS 3D 数据/'
    :return:
    """
    print("------create similar couple------")
    folder = np.random.choice(glob.glob(file_path + "*"))
    depth_file = np.random.choice(glob.glob(folder + "/*.tif"))
    if depth_file in list_similar_path:
        return create_similar_couple(file_path)
    else:
        list_similar_path.append(depth_file)
    print(depth_file)
    mat = recon(depth_file)
    # print(mat.shape)
    mat_similar = recon(depth_file, similar=True)
    floor = mat.shape[0]

    similar_couple = []
    for m in (mat, mat_similar):
        full = np.zeros(shape=shape)
        full[:floor, :, :] = m[:, :shape[1], :shape[2]]
        similar_couple.append(full)

    print("-----create similar finished------")
    return np.array(similar_couple)


def create_wrong_couple(file_path):
    """
    不同样本对
    :param file_path: './NUS 3D 数据/'
    :return:
    """
    print("------create wrong couple------")
    wrong_couple = []
    temp_path = None
    for i in range(2):
        folder = np.random.choice(glob.glob(file_path + "*"))
        depth_file = np.random.choice(glob.glob(folder + "/*.tif"))
        if i:
            if temp_path == depth_file:
                return create_wrong_couple(file_path)
            if temp_path in dict_wrong_path:
                if depth_file in dict_wrong_path[temp_path]:
                    return create_wrong_couple(file_path)
                else:
                    dict_wrong_path[temp_path].add(depth_file)
            else:
                dict_wrong_path[temp_path] = set()
        print(depth_file)
        mat = recon(depth_file)
        # print(mat.shape)

        floor = mat.shape[0]
        full = np.zeros(shape=shape)
        full[:floor, :, :] = mat[:, :shape[1], :shape[2]]

        wrong_couple.append(full)
        temp_path = depth_file
    print("------create wrong finished------")
    return np.array(wrong_couple)


""""
name: pytorch-env
 pytorch 1.2.0

dependencies:
  - python=3.6
  - matplotlib=2.2.2
  - jupyter=1.0
  - torchvision 0.4.0
"""

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary


# Custom Dataset Function¶
# This dataset generates a pair of images. 0 for genuine pair and 1 for impostor pair
def generator(batch_size):
    while 1:
        X = []
        y = []
        switch = True
        for _ in range(batch_size):
            #   switch += 1
            if switch:
                #   print("correct")
                X.append(create_similar_couple(path).reshape((2, shape[0], shape[1], shape[2])))
                y.append(np.array([0.]))
            else:
                #   print("wrong")
                X.append(create_wrong_couple(path).reshape((2, shape[0], shape[1], shape[2])))
                y.append(np.array([1.]))
            switch = not switch

        X = np.asarray(X)
        y = np.asarray(y)

        print("------next------")
        # yield X[:, 0], X[:, 1], y
        return X[:, 0], X[:, 1], y


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, data):
        self.x1_data = torch.from_numpy(data[0])
        self.x2_data = torch.from_numpy(data[1])
        self.y_data = torch.from_numpy(data[2])
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x1_data[index], self.x2_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# Neural Net Definition
# We will use a standard convolutional neural network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(shape[0], 64, kernel_size=4, stride=2),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=4, stride=2),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, kernel_size=4, stride=2),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 91 * 91, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# Contrastive Loss¶
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


if __name__ == "__main__":

    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    def show_plot(iteration, loss):
        plt.plot(iteration, loss)
        plt.savefig("fig_loss_trend.png")
        plt.show()


    # Training Time!
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    net = SiameseNetwork().cpu()
    print(summary(net, input_size=[shape, shape], device="cpu"))

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.2)

    batch_size = 22
    siamese_dataset = generator(batch_size)
    train_dataloader = DataLoader(DealDataset(siamese_dataset),
                                  shuffle=True,
                                  num_workers=10,
                                  batch_size=1)
    siamese_dataset = 0
    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, 5):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.double(), img1.double(), label.double()
            # img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            net = net.double()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    torch.save(net, 'model_cpu.pkl')
    show_plot(counter, loss_history)
