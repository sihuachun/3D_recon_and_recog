# coding: utf-8
import numpy as np
import warnings
import os
import glob
import cv2
import random
import paddle.fluid as fluid
from paddle.fluid import Layer
from paddle.fluid.dygraph import nn
from paddle.fluid import layers
from paddle.fluid.contrib.model_stat import summary
from paddle.utils.plot import Ploter
import datetime

t = datetime.datetime.now()
t = t.strftime('%Y-%m-%d-%H')
shape = (3, 728, 728)
warnings.filterwarnings("ignore")


# Configuration Class
# A simple class to manage configuration
class Config():
    training_dir = "/home/aistudio/work/datasets/Transform_2D"
    # training_dir = "./Transform_2D"
    train_batch_size = 8
    train_number_epochs = 10

    @staticmethod
    def create_data_couple():
        data_couple = []
        folders = os.listdir(Config.training_dir)
        paths = []
        for folder in folders:
            path = os.path.join(Config.training_dir, folder)
            sub_path = glob.glob(path+'/*.jpg')
            sub_path = sorted(sub_path)
            paths += [os.path.join(path, _) for _ in sub_path]
        # 构建数据集地址对
        for f in range(len(paths)):
            for s in range(f + 1, f + 5):
                if f+10 < len(paths):
                    # print((paths[f], paths[s]))
                    data_couple.append((paths[f], paths[s]))
        random.shuffle(data_couple)
        print(len(data_couple))
        return data_couple


# Custom Dataset Class¶
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
class Dataset():
    def my_transform(self, Objects):
        """
        :param Objects: NHWC format
        :return: Objects: NCHW format
        """
        full = []
        for Object in Objects:
            Object = cv2.cvtColor(Object, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
            Object = np.transpose(Object, axes=[2, 0, 1]).copy()
            Object = np.resize(Object, shape)
            full.append(Object)
        return np.array(full)

    def generate(self, data_couple):
        Input1, Input2, Label = [], [], []
        for _ in range(batch):
            f_path, s_path = data_couple.pop()
            path1, path2 = f_path.split('/')[-1], s_path.split('/')[-1]
            if path1.split('_')[0] == path2.split('_')[0]:
                label = np.array([0.0])
            else:
                label = np.array([1.0])

            if f_path == s_path:
                ob0 = ob1 = cv2.imread(f_path)
            else:
                ob0 = cv2.imread(f_path)
                ob1 = cv2.imread(s_path)
            Input1.append(ob0)
            Input2.append(ob1)
            Label.append(label)
            # print(f_path, s_path, label)

        img0 = self.my_transform(Input1)
        img1 = self.my_transform(Input2)
        return img0, img1, np.array(Label)


# Neural Net Definition
# We will use a standard convolutional neural network
class Block(Layer):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, in_planes, filter_size=3, stride=stride, padding=1, groups=in_planes,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm(in_planes)
        self.conv2 = nn.Conv2D(in_planes, out_planes, filter_size=1, stride=1, padding=0, bias_attr=False)
        self.bn2 = nn.BatchNorm(out_planes)

    def forward(self, x):
        out = layers.relu(self.bn1(self.conv1(x)))
        out = layers.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(Layer):
    # (128,2) means conv planes=128, conv stride=2,
    # by default conv stride=1
    cfg = [(64, 2), 64, (128, 2), 128, (256, 2), 256, (256, 2), 256, (256, 2),
           256, (512, 2), 512, (512, 2), 512, (512, 2), 512, (1024, 2), 1024]

    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2D(shape[0], 32, filter_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm(32)
        self._layers = self._make_layers(in_planes=32)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return fluid.dygraph.Sequential(*layers)

    def forward(self, x):
        out = layers.relu(self.bn1(self.conv1(x)))
        out = self._layers(out)
        out = layers.pool2d(out, pool_stride=2, pool_type='avg', pool_size=3, pool_padding=1)
        out = layers.reshape(out, (out.shape[0], -1))
        return out


class SiameseNetwork(fluid.dygraph.Layer):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = MobileNet()
        self.linear = fluid.dygraph.Sequential(
            nn.Linear(input_dim=1024, output_dim=512, act='relu'),
            nn.Linear(input_dim=512, output_dim=256, act='relu'),
            nn.Linear(input_dim=256, output_dim=128))

    def forward(self, input1, input2):
        t1, t2 = self.model(input1), self.model(input2)
        m1, m2 = self.linear(t1), self.linear(t2)
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
        loss_contrastive = layers.reduce_mean(
            layers.elementwise_mul(1 - label, layers.square(euclidean_distance), axis=0) +
            layers.elementwise_mul(label, layers.square(layers.clamp(self.margin - euclidean_distance, min=0.0)),
                                   axis=0), dim=1)

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
        param_dict, opt_dict = fluid.load_dygraph('./checkpoint/Transform_2D/transform_2D_epoch10')
        model.load_dict(param_dict)
        model.train()
        all_data_couple = Config.create_data_couple()
        # 定义学习率，并加载优化器参数到模型中
        total_steps = len(all_data_couple) * (Config.train_number_epochs - 1)
        # lr = fluid.dygraph.PolynomialDecay(0.01, total_steps, 0.0001)
        lr = 0.0001
        # 定义优化器
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=model.parameters(),
                                                  regularization=fluid.regularizer.L2Decay(regularization_coeff=0.1))
        # 定义损失函数
        loss_contrastive = ContrastiveLoss(30)
        # Training Time!
        train_prompt = "Train cost"
        plot_cost = Ploter(train_prompt)
        cur_loss = 0
        for epoch in range(1, Config.train_number_epochs + 1):
            step = 0
            data_couple = all_data_couple.copy()
            sum_loss = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.0)
            while len(data_couple) >= batch:
                siamese_dataset = Dataset().generate(data_couple)
                Input1, Input2, Label = [fluid.dygraph.to_variable(x.astype('float32')) for x in siamese_dataset]
                Output1, Output2 = model(Input1, Input2)
                loss, distance, label = loss_contrastive.forward(Output1, Output2, Label)
                avg_loss = layers.reduce_mean(loss)
                sum_loss += avg_loss
                # 后向传播，更新参数的过程
                loss.backward()
                optimizer.minimize(loss)
                model.clear_gradients()
                print(
                    "Epoch number {}, step {}: Label {}; Euclidean Distance {}; Current loss {}\n".format(epoch, step, np.concatenate(label), np.concatenate(distance),
                                                                                                 avg_loss.numpy()))
                step += 1
            plot_cost.append(train_prompt, epoch, sum_loss.numpy()/step)

            # 保存模型目录
            save_path = './checkpoint/' + Config.training_dir.split('/')[-1]
            os.makedirs(save_path, exist_ok=True)
            # 保存模型参数
            save_path += '/transform_2D_epoch{}'.format(epoch)
            fluid.save_dygraph(model.state_dict(), save_path)
            fluid.save_dygraph(optimizer.state_dict(), save_path)
            cur_loss = sum_loss.numpy()
        os.makedirs('/home/aistudio/figure_and_csv', exist_ok=True)
        plot_cost.plot('/home/aistudio/figure_and_csv/' + Config.training_dir.split('/')[-1] + '_{}.png'.format(t))
