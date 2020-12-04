# coding: utf-8
import numpy as np
import warnings
import os
import glob
import tifffile
import random
import paddle.fluid as fluid
from paddle.fluid import Layer
from paddle.fluid.dygraph import nn
from paddle.fluid import layers
from paddle.fluid.contrib.model_stat import summary
from paddle.utils.plot import Ploter
import datetime
import pandas as pd
import cv2

t = datetime.datetime.now()
t = t.strftime('%Y-%m-%d-%H')
shape = (71, 512, 512)
warnings.filterwarnings("ignore")


def leaky_relu(x):
    return layers.leaky_relu(x, alpha=0.1, name=None)


# Configuration Class
# A simple class to manage configuration
class Config():
    training_dir = "/home/aistudio/datasets/3D/train"

    @staticmethod
    def create_data_couple():
        data_couple = []
        paths = glob.glob(Config.training_dir + "/*.tif")
        # 构建数据集地址对
        for f in range(len(paths)):
            for s in range(f, len(paths)):
                data_couple.append((paths[f], paths[s]))
        print(len(data_couple))
        return data_couple


# Custom Dataset Class¶
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
class Dataset():
    def my_transform(self, *Objects):
        """
        resize Object.shape to a fixed shape
        """
        full = []
        for i in range(len(Objects)):
            Object = Objects[i]
            # if Object.shape[1:] != shape[1:]:
            #     Object = np.transpose(Object, (1, 2, 0))
            #     Object = cv2.resize(Object, shape[1:], interpolation=cv2.INTER_AREA)
            #     Object = np.transpose(Object, (2, 0, 1))
            # if Object.shape[0] > shape[0]:
            #     Object = Object[:shape[0], ...]
            # if Object.shape[0] < shape[0]:
            #     module = np.zeros(shape)
            #     module[:Object.shape[0], ...] = Object
            #     Object = module
            full.append(Object)
        return np.array(full)

    def generate(self, data_couple):
        f_path, s_path = data_couple.pop()
        if f_path == s_path:
            label = np.array([0.0])
        else:
            label = np.array([1.0])

        if f_path == s_path:
            ob0 = ob1 = tifffile.imread(f_path)
        else:
            ob0 = tifffile.imread(f_path)
            ob1 = tifffile.imread(s_path)

        # print(f_path, s_path, label)
        img0 = self.my_transform(ob0)
        img1 = self.my_transform(ob1)
        return f_path, s_path, img0, img1, label


# Neural Net Definition
# We will use a standard convolutional neural network
class Block(Layer):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, in_planes, filter_size=3, stride=stride, padding=1, groups=in_planes,
                               bias_attr=False)
        self.bn1 = nn.InstanceNorm(in_planes)
        self.conv2 = nn.Conv2D(in_planes, out_planes, filter_size=1, stride=1, padding=0)
        self.bn2 = nn.InstanceNorm(out_planes)

    def forward(self, x):
        out = leaky_relu(self.bn1(self.conv1(x)))
        out = leaky_relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(Layer):
    # (128,2) means conv planes=128, conv stride=2,
    # by default conv stride=1
    channel = shape[0]
    cfg = [channel, (channel, 2), channel, (channel, 2), channel, (channel, 2), channel, (channel, 2), channel,
           (channel, 2), channel, (channel, 2)]

    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2D(shape[0], shape[0], filter_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.InstanceNorm(shape[0])
        self._layers = self._make_layers(in_planes=shape[0])
        self.conv_last = nn.Conv2D(shape[0], shape[0], filter_size=3, stride=1, padding=1, bias_attr=False)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return fluid.dygraph.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self._layers(out)
        out = self.conv_last(out)
        out = layers.reshape(out, (out.shape[0], -1))
        return out


class SiameseNetwork(fluid.dygraph.Layer):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = MobileNet()
        self.linear1 = nn.Linear(input_dim=71 * 8 * 8, output_dim=71 * 8 * 4)
        self.linear2 = nn.Linear(input_dim=71 * 8 * 4, output_dim=71 * 8 * 2)
        self.linear3 = nn.Linear(input_dim=71 * 8 * 2, output_dim=568)

    def _linear(self, x):
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def forward(self, input1, input2):
        t1, t2 = self.model(input1), self.model(input2)
        m1, m2 = self._linear(t1), self._linear(t2)
        return m1, m2


# Contrastive Loss¶
class ContrastiveLoss(Layer):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        :param output1: [n, 128]
        :param output2: [n, 128]
        :param label: [n, 1]
        :return: [1]
        """
        distance = layers.elementwise_sub(output1, output2)
        distance = layers.square(distance)
        euclidean_distance = layers.reduce_sum(distance, dim=1, keep_dim=True)
        euclidean_distance = layers.sqrt(euclidean_distance)
        loss_contrastive = layers.elementwise_mul(1 - label, layers.square(euclidean_distance),
                                                  axis=0) + layers.elementwise_mul(label, layers.square(
            layers.clamp(self.margin - euclidean_distance, min=0.0)), axis=0)
        return loss_contrastive, euclidean_distance.numpy(), label.numpy()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    df = pd.DataFrame(columns=["path_sample1", "path_sample2", "label", "distance"])
    # 定义飞桨动态图工作环境
    use_gpu = True
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # 声明网络结构
        model = SiameseNetwork()
        # 保存模型目录
        model_path = './checkpoint/3D/transform_3D_epoch94'
        # 加载模型参数
        param_dict, _ = fluid.load_dygraph(model_path)
        model.load_dict(param_dict)
        model.eval()
        # 定义损失函数
        loss_contrastive = ContrastiveLoss(512)
        data_couple = Config.create_data_couple()
        # Training Time!
        for _ in range(len(data_couple)):
            siamese_dataset = Dataset().generate(data_couple)
            Input1, Input2, Label = [fluid.dygraph.to_variable(x.astype('float32')) for x in siamese_dataset[2:]]
            Output1, Output2 = model(Input1, Input2)
            loss, distance, label = loss_contrastive.forward(Output1, Output2, Label)
            print("Label {}; Euclidean distance {}; Current loss {}\n".format(label, distance, loss.numpy()))
            df = df.append(
                {"path_sample1": siamese_dataset[0], "path_sample2": siamese_dataset[1], "label": label[0],
                 "distance": distance[0][0]}, ignore_index=True)
    csv_path = '/home/aistudio/figure_and_csv/' + Config.training_dir.split('/')[-1] + '_3D.csv'
    df.to_csv(csv_path, index=False)
