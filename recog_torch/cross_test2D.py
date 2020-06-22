"""
2D data cross test
"""

# Imports
# All the imports are defined her
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import torch
import PIL.ImageOps
import torch.nn as nn
import torch.nn.functional as F
import glob

shape = (1, 100, 100)
path = "./data2D/training"


# Custom Dataset Class¶
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
class SiameseNetworkDataset(Dataset):

    def __init__(self, f_path, r_path, transform=None, should_invert=True):
        self.f_path = f_path
        self.r_path = r_path
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):

        img0 = Image.open(self.f_path)
        img1 = Image.open(self.r_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        # print(img0.shape)  # (1, 100, 100)

        return img0, img1, torch.from_numpy(
            np.array([int(self.f_path.split('\\')[-2] != self.r_path.split('\\')[-2])], dtype=np.float32))

    def __len__(self):
        return 1


# Neural Net Definition
# We will use a standard convolutional neural network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 512),
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
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def cross_test_all(file_path):
    """
    交叉验证输出所有样本对之间的相似度， distance 越大，相似度越低。
    :param file_path: './NUS 3D 数据/'
    :return:
    """

    distance_all = {}
    criterion = ContrastiveLoss()
    file_path_smaple = glob.glob(file_path + "/*")
    file_path_all = []
    for path_smaple in file_path_smaple:
        file_path_all += glob.glob(path_smaple + "/*.jpg")
    print(file_path_all)
    length_path_all = len(file_path_all)

    for f in range(length_path_all):
        f_path = file_path_all[f]
        for r in range(f+1, length_path_all):
            r_path = file_path_all[r]

            siamese_dataset = SiameseNetworkDataset(f_path=f_path, r_path=r_path,
                                                    transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                                  transforms.ToTensor()])
                                                    , should_invert=False)

            test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)

            siamese_dataset = None
            dataiter = iter(test_dataloader)
            img0, img1, label = next(dataiter)
            test_dataloader, dataiter = None, None
            # print(img0.shape, img0.dtype)
            output1, output2 = model(img0, img1)
            loss = criterion(output1, output2, label)
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            output1, output2 = None, None
            print(f_path, r_path)
            print("label: {}, distance: {}, loss: {}".format(label.item(), euclidean_distance.item(), loss))
            key = str((f_path, r_path))
            distance_all[key] = {"label": label.item(), "distance": euclidean_distance.item(), "loss": loss.item()}
            label, euclidean_distance = None, None

    print(distance_all)
    return distance_all


if __name__ == "__main__":

    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model = torch.load('model_2D.pkl')
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
    df.to_csv("cross2D_{}.csv".format(test_name), index=False)
