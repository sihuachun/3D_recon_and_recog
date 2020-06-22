"""
3D data cross test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import dxchange
import tomopy
import numpy as np
import glob
import warnings

path = './NUS 3D 数据/'
shape = (100, 728, 728)
warnings.filterwarnings("ignore")


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


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, data):
        self.x1_data = torch.from_numpy(data[0])
        self.x2_data = torch.from_numpy(data[1])
        self.y_data = torch.from_numpy(data[2])
        self.len = self.y_data.shape[0]

    def __getitem__(self, index=0):
        return self.x1_data, self.x2_data, self.y_data

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
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, kernel_size=4, stride=2),
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


def cross_test_all(file_path):
    """
    交叉验证输出所有样本对之间的相似度， distance 越大，相似度越低。
    :param file_path: './NUS 3D 数据/'
    :return:
    """
    distance_all = {}
    criterion = ContrastiveLoss()
    file_path_smaple = glob.glob(file_path + "*")
    file_path_all = []
    for path_smaple in file_path_smaple:
        file_path_all += glob.glob(path_smaple + "/*.tif")

    length_path_all = len(file_path_all)
    for f in range(length_path_all):
        p = file_path_all[f]
        key = str((p, p))
        # if key not in crossed_dict:

        print("------create similar couple: {}------".format(key))
        mat = recon(p)
        mat_similar = recon(p, similar=True)

        similar_couple = []
        floor = mat.shape[0]
        for m in (mat, mat_similar):
            full = np.zeros(shape=shape)
            full[:floor, :, :] = m[:, :shape[1], :shape[2]]
            similar_couple.append(full)
        similar_couple = [np.array(similar_couple).reshape((2, shape[0], shape[1], shape[2]))]
        similar_couple = np.asarray(similar_couple)
        similar_couple = (similar_couple[:, 0], similar_couple[:, 1], np.array([0.0]))
        print("-----create similar finished------")
        data = DealDataset(similar_couple).__getitem__()
        similar_couple = None
        img0, img1, label = data
        img0, img1, label = img0.double(), img1.double(), label.double()
        output1, output2 = model(img0, img1)
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        los = criterion(output1, output2, label).item()
        print("distance: {}, loss: {}".format(euclidean_distance.item(), los))
        distance_all[key] = {"distance": euclidean_distance.item(), "loss": los}
        # else:
        #     print("------crossed sample couple------")

        # break
        for r in range(f + 1, length_path_all):
            p2 = file_path_all[r]
            key = str((p, p2))
            # if key not in crossed_dict:
            print("------create wrong couple: {}------".format(key))
            mat = recon(p)
            mat_wrong = recon(p2)

            wrong_couple = []
            for m in (mat, mat_wrong):
                floor = m.shape[0]
                full = np.zeros(shape=shape)
                full[:floor, :, :] = m[:, :shape[1], :shape[2]]
                wrong_couple.append(full)
            wrong_couple = [np.array(wrong_couple).reshape((2, shape[0], shape[1], shape[2]))]
            wrong_couple = np.asarray(wrong_couple)
            wrong_couple = (wrong_couple[:, 0], wrong_couple[:, 1], np.array([1.0]))
            print("------create wrong finished------")
            data = DealDataset(wrong_couple).__getitem__()
            wrong_couple = None
            img0, img1, label = data
            img0, img1, label = img0.double(), img1.double(), label.double()
            output1, output2 = model(img0, img1)
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            los = criterion(output1, output2, label).item()
            print("distance: {}, loss: {}".format(euclidean_distance.item(), los))
            distance_all[key] = {"distance": euclidean_distance.item(), "loss": los}
            # else:
            #     print("------crossed sample couple------")
    print(distance_all)
    return distance_all


if __name__ == "__main__":

    model = torch.load('model_3D.pkl')
    model.eval()
    da = cross_test_all(file_path=path)
    # da.update(crossed_dict)
    del model

    import pandas as pd

    df = pd.DataFrame(columns=["path_sample1", "path_sample2", "distance", "loss"])
    for k in da:
        p1, p2 = k.split(", ")
        df = df.append(
            {"path_sample1": p1[1:], "path_sample2": p2[:-1], "distance": da[k]['distance'], "loss": da[k]['loss']},
            ignore_index=True)
    df.to_csv("cross3D.csv", index=False)
