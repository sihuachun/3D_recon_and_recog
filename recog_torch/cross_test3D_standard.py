import dxchange
import tomopy
import numpy as np
import glob
import warnings
from torch.utils.data import DataLoader, Dataset

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


# Custom Dataset Class¶
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
class SiameseNetworkDataset(Dataset):

    def __init__(self, f_path, s_path):
        self.f_path, self.s_path = f_path, s_path
        self.data_X1, self.data_X2, self.data_y = self.generate()

    def my_transform(self, Object):
        floor = Object.shape[0]
        full = np.zeros(shape)
        full[:floor] = Object
        return torch.from_numpy(full)

    def generate(self):

        img0 = recon(self.f_path)
        img1 = recon(self.s_path)

        img0 = self.my_transform(img0)
        img1 = self.my_transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(self.f_path != self.s_path)], dtype=np.float32))

    def __getitem__(self, index=None):
        return self.data_X1, self.data_X2, self.data_y

    def __len__(self):
        return 1


import random
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def cross_test_all(file_path):
    """
    交叉验证输出所有样本对之间的相似度， distance 越大，相似度越低。
    :param file_path: './sample1_standard_transform/'
    :return:
    """

    distance_all = {}
    criterion = ContrastiveLoss(6)
    # file_path_smaple = glob.glob(file_path + "/*")
    # file_path_all = []
    # for path_smaple in file_path_smaple:
    #     file_path_all += glob.glob(path_smaple + "/*.tif")

    file_path_all = glob.glob(file_path + "/*.tif")
    print(len(file_path_all))
    print(file_path_all)
    length_path_all = len(file_path_all)

    for f in range(length_path_all):
        f_path = file_path_all[f]
        for r in range(f+1, length_path_all):
            r_path = file_path_all[r]

            siamese_dataset = SiameseNetworkDataset(f_path=f_path, s_path=r_path)

            test_dataloader = DataLoader(siamese_dataset, num_workers=8, batch_size=1, shuffle=True)

            dataiter = iter(test_dataloader)
            img0, img1, label = next(dataiter)
            img0, img1 = img0.float(), img1.float()
            # print(img0.shape, img0.dtype)
            output1, output2 = model(img0, img1)
            loss = criterion(output1, output2, label)
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            print(f_path, r_path)
            print("label: {}, distance: {}, loss: {}".format(label.item(), euclidean_distance.item(), loss))
            key = str((f_path, r_path))
            distance_all[key] = {"label": label.item(), "distance": euclidean_distance.item(), "loss": loss.item()}

    print(distance_all)
    return distance_all


if __name__ == "__main__":

    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    path = "./NUS 3D 数据/sample1_standard"
    model = SiameseNetwork().float()
    check_point = torch.load('model_3D_standard.pkl', map_location=torch.device('cpu'))
    model.load_state_dict(check_point)
    model.eval()
    da = cross_test_all(file_path=path)
    del model

    import pandas as pd

    df = pd.DataFrame(columns=["path_sample1", "path_sample2", "label", "distance", "loss"])
    for k in da:
        p1, p2 = k.split(", ")
        df = df.append(
            {"path_sample1": p1[1:], "path_sample2": p2[:-1], "label": da[k]['label'], "distance": da[k]['distance'],
             "loss": da[k]['loss']},
            ignore_index=True)
    test_name = path.split('/')[-1]
    df.to_csv("cross3D_standard_{}.csv".format(test_name), index=False)

    # # 可以考虑直接存储为方便画heatMap的数据格式，可以参考这段代码
    # heatMapData = np.zeros((FinalX.shape[0], FinalX.shape[0]))
    #
    # import time
    #
    # t1 = time.time()
    # for i in range(FinalX.shape[0]):
    #     for j in range(FinalX.shape[0]):
    #         ssim_noise = ssim(FinalX[i], FinalX[j], multichannel=True)
    #         print(i, j, ssim_noise)
    #         print('Running time: ', time.time() - t1)
    #         heatMapData[i, j] = ssim_noise
    #         t1 = time.time()
    #
    # np.save("heatmapData_1222_296.npy", heatMapData)
