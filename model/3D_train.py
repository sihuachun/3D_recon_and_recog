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
from paddle.utils.plot import Ploter
import datetime
import cv2

t = datetime.datetime.now()
t = t.strftime('%Y-%m-%d-%H')
# 训练集数据最大SHAPE
shape = (71, 512, 512)
warnings.filterwarnings("ignore")


def leaky_relu(x):
    return layers.leaky_relu(x, alpha=0.1, name=None)


# Configuration Class
# A simple class to manage configuration
class Config():
    training_dir = "/home/aistudio/datasets/3D"
    # training_dir = "./Transform_3D"
    train_batch_size = 1
    train_number_epochs = 60

    @staticmethod
    def create_data_couple():
        """构建数据集地址对"""
        data_couple = []
        # # 正样本对数据集
        same_path = glob.glob('/home/aistudio/datasets/3D/same/*')
        for path in same_path:
            sub_path = glob.glob(path + '/*.tif')
            for f in range(len(sub_path)):
                for s in range(f + 1, len(sub_path)):
                    data_couple.append((sub_path[f], sub_path[s]))
        print("Same dataset lenght: {}".format(len(data_couple)))
        # 负样本对数据集
        paths = glob.glob(Config.training_dir + '/train/*.tif')
        for f in range(len(paths)):
            for s in range(f + 1, len(paths)):
                # print((paths[f], paths[s]))
                data_couple.append((paths[f], paths[s]))
        random.shuffle(data_couple)
        print("All dataset lenght: {}".format(len(data_couple)))
        return data_couple


# Custom Dataset Class¶
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
class Dataset():
    def my_transform(self, Objects):
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

    def generate(self, data_couple, batch):
        Input1, Input2, Label = [], [], []
        for _ in range(batch):
            f_path, s_path = data_couple.pop()
            path1, path2 = f_path.split('/')[-1], s_path.split('/')[-1]
            if path1.split('_')[0] == path2.split('_')[0]:
                label = np.array([0.0])
            else:
                label = np.array([1.0])
            ob0 = tifffile.imread(f_path)
            ob1 = tifffile.imread(s_path)
            Input1.append(ob0.astype(np.float32))
            Input2.append(ob1.astype(np.float32))
            Label.append(label)
            # print(f_path, s_path, label)
        img0 = self.my_transform(Input1)
        img1 = self.my_transform(Input2)
        # print(img0.shape, img1.shape)
        return img0, img1, np.array(Label)


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
        self.linear1 = nn.Linear(input_dim=71*8*8, output_dim=71*8*4)
        self.linear2 = nn.Linear(input_dim=71*8*4, output_dim=71*8*2)
        self.linear3 = nn.Linear(input_dim=71*8*2, output_dim=568)

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
    batch = Config.train_batch_size
    # 定义飞桨动态图工作环境
    use_gpu = True
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # 声明网络结构
        model = SiameseNetwork()
        # 恢复训练
        # param_dict, _ = fluid.load_dygraph('./checkpoint/3D/transform_3D_epoch0')
        # model.load_dict(param_dict)
        model.train()
        # 定义学习率，并加载优化器参数到模型中
        # total_steps = len(train_data_couple) * (Config.train_number_epochs - 1)
        # lr = fluid.dygraph.PolynomialDecay(0.01, total_steps, 0.0001)
        # 定义优化器
        # optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=model.parameters())
        # regularization=fluid.regularizer.L2Decay(regularization_coeff=0.1))
        optimizer = fluid.optimizer.SGD(learning_rate=0.000001, parameter_list=model.parameters())
        # 定义损失函数
        loss_contrastive = ContrastiveLoss(660)
        # Training Time!
        train_prompt = "Train cost"
        train_cost = Ploter(train_prompt)
        print("Training time ...")
        _loss = float("INF")
        for epoch in range(1, Config.train_number_epochs + 1):
            """Training"""
            step = 0
            data_couple = Config.create_data_couple()
            sum_loss = 0
            while len(data_couple) >= batch:
                siamese_dataset = Dataset().generate(data_couple, batch=Config.train_batch_size)
                Input1, Input2, Label = [fluid.dygraph.to_variable(x.astype('float32')) for x in siamese_dataset]
                # print(Input1.shape, Input2.shape, Label.shape)
                Output1, Output2 = model(Input1, Input2)
                loss, distance, label = loss_contrastive.forward(Output1, Output2, Label)
                avg_loss = layers.reduce_mean(loss)
                sum_loss += avg_loss.numpy()[0]
                print("Epoch number {}, step {}: Label {}; Euclidean Distance {}; Current loss {}\n".
                      format(epoch, step, np.concatenate(label), np.concatenate(distance), avg_loss.numpy()))
                # 后向传播，更新参数的过程
                # if distance == 0:
                #     continue
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()
                step += 1
            train_cost.append(train_prompt, epoch, sum_loss / step)

            # 保存模型目录
            save_path = './checkpoint/' + Config.training_dir.split('/')[-1]
            os.makedirs(save_path, exist_ok=True)
            # 保存模型参数
            if sum_loss < _loss:
                _loss = sum_loss
                save_path += '/transform_3D_epoch{}'.format(epoch)
                fluid.save_dygraph(model.state_dict(), save_path)
                # fluid.save_dygraph(optimizer.state_dict(), save_path)

        os.makedirs('/home/aistudio/figure_and_csv', exist_ok=True)
        train_cost.plot(
            '/home/aistudio/figure_and_csv/' + Config.training_dir.split('/')[-1] + '_train_{}.png'.format(t))
