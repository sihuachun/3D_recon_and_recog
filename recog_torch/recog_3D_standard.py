import dxchange
import tomopy
import numpy as np
import glob
import warnings
from torch.utils.data import Dataset
import glob

shape = (100, 728, 728)
warnings.filterwarnings("ignore")


def recon(path):
    """
    reconstruction object
    :param similar: boolean, specify to or not to create reconstructed objedct
    :param path: image location
    :return: 3-D object
    """

    obj = dxchange.reader.read_tiff(fname=path, slc=None)  # Generate an object

    ang = tomopy.angles(180)  # Generate uniformly spaced tilt angles

    obj = obj.astype(np.float32)
    # print(obj.shape)

    sim = tomopy.project(obj, ang)  # Calculate projections
    rec = tomopy.recon(sim, ang, algorithm='art')  # Reconstruct object
    # print(rec.shape)

    return rec


# Configuration Class
# A simple class to manage configuration
class Config():
    training_dir = "./NUS 3D 数据/sample1_standard/"
    # testing_dir = "./sample1_standard_transform/testing/"
    train_batch_size = 16

    def __init__(self, all_data_couple=None):
        r = len(all_data_couple) % self.train_batch_size
        self.train_number_epochs = len(all_data_couple) // self.train_batch_size + 1 if r \
            else len(all_data_couple) // self.train_batch_size


# Custom Dataset Class¶
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
class SiameseNetworkDataset(Dataset):

    def __init__(self):
        self.data_X, self.data_y = self.generate()

    def my_transform(self, Object):
        floor = Object.shape[0]
        full = np.zeros(shape)
        full[:floor] = Object
        return torch.from_numpy(full)

    def generate(self):
        X, y = [], []
        for i in range(Config.train_batch_size):
            if not all_data_couple:
                break

            f_path, s_path = all_data_couple.pop()
            ob0 = recon(f_path)
            ob1 = recon(s_path)

            img0 = self.my_transform(ob0)
            img1 = self.my_transform(ob1)

            X.append((img0, img1))
            # label = torch.from_numpy(np.array([int(f_path.split('\\')[-2] != s_path.split('\\')[-2])], dtype=np.float32))
            label = torch.from_numpy(np.array([int(f_path != s_path)], dtype=np.float32))
            y.append(label)

            print(f_path, s_path, label.item())

        return X, y

    def __getitem__(self, index):
        return self.data_X[index][0], self.data_X[index][1], self.data_y[index]

    def __len__(self):
        return len(self.data_y)


import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary


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
        plt.savefig("fig_loss_trend_3D_standard.png")
        plt.show()


    # file_path_smaple = glob.glob(Config.training_dir + "*")
    # file_path_all = []
    # for path_smaple in file_path_smaple:
    #     file_path_all += glob.glob(path_smaple + "/*.tif")
    # print(file_path_all)
    file_path_all = glob.glob(Config.training_dir + '*.tif')
    print(len(file_path_all))
    all_data_couple = set()
    for f in range(len(file_path_all)):
        for s in range(f, len(file_path_all)):
            all_data_couple.add((file_path_all[f], file_path_all[s]))
    print(all_data_couple)
    print(len(all_data_couple))
    # Training Time!
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    net = SiameseNetwork().cpu()
    print(summary(net, input_size=[shape, shape], device="cpu"))
    net = net.float()

    criterion = ContrastiveLoss(6)
    optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.2)

    # epoch 0
    counter = []
    loss_history = []
    iteration_number = 0
    i_counter = 0
    for _ in range(Config(all_data_couple).train_number_epochs):

        # Training Time!
        siamese_dataset = SiameseNetworkDataset()
        train_dataloader = DataLoader(siamese_dataset,
                                      shuffle=True,
                                      num_workers=8,
                                      batch_size=1)

        for epoch in range(0, 2):
            for i, data in enumerate(train_dataloader, 0):
                img0, img1, label = data
                img0, img1 = img0.float(), img1.float()
                optimizer.zero_grad()
                output1, output2 = net(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                loss_contrastive.backward()
                optimizer.step()
                if i_counter % 10 == 0:
                    print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                    iteration_number += 10
                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())
                i_counter += 1


    torch.save(net.state_dict(), 'model_3D_standard.pkl')
    show_plot(counter, loss_history)
