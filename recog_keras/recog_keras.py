#!/usr/bin/env python 3.6.7
# -*- coding: utf-8 -*-
# author  : sihuachun
# date    : 2020/6/12
# software: PyCharm


import dxchange
import tomopy
import numpy as np
import glob
import warnings
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

path = './NUS 3D 数据/'
shape = (80, 728, 728)
warnings.filterwarnings("ignore")
data_format = 'channels_first'
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
                    dict_wrong_path[temp_path] += {file_path}
            else:
                dict_wrong_path[temp_path] = {}
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


"""# Network crafting."""
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Input, \
    BatchNormalization, concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras import backend as K


def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    margin = 2.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))


def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1, 1), padding='valid', data_format=data_format)(x)
    x = Activation('relu')(x)

    left = Convolution2D(expand, (1, 1), padding='valid', data_format=data_format)(x)
    left = Activation('relu')(left)

    right = Convolution2D(expand, (1, 1), padding='valid', data_format=data_format)(x)
    right = Activation('relu')(right)

    x = concatenate([left, right], axis=1)
    return x


def generator(batch_size):
    while 1:
        X = []
        y = []
        switch = True
        for _ in range(batch_size):
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
        # yield [X[:, 0], X[:, 1]], y
        return [X[:, 0], X[:, 1]], y


def create_network():
    """
    create neural network
    :return: needed siamese network
    """
    """# model_1"""
    img_input = Input(shape=shape)
    x = Convolution2D(64, kernel_size=(5, 5), strides=(2, 2), padding='valid', data_format=data_format)(img_input)
    x = BatchNormalization(axis=1, scale=True, trainable=True)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(x)

    x = fire(x, squeeze=16, expand=16)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(x)

    x = fire(x, squeeze=32, expand=32)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(x)

    x = fire(x, squeeze=64, expand=64)

    x = Dropout(0.2)(x)

    # x = Convolution2D(512, kernel_size=(1, 1), padding='same', data_format=data_format)(x)

    out = Activation('relu')(x)

    # modelsqueeze = Model(img_input, out)
    #
    # modelsqueeze.summary()
    #
    # """# model_2"""
    # im_in = Input(shape=shape)
    #
    # x1 = modelsqueeze(im_in)
    #
    # """
    # x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
    # x1 = Dropout(0.4)(x1)
    #
    # x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x1)
    #
    # x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
    # x1 = BatchNormalization()(x1)
    # x1 = Dropout(0.4)(x1)
    #
    # x1 = Convolution2D(64, (1,1), padding='same', activation="relu")(x1)
    # x1 = BatchNormalization()(x1)
    # x1 = Dropout(0.4)(x1)
    # """

    x1 = Flatten()(out)

    x1 = Dense(512, activation="relu")(x1)
    x1 = Dropout(0.2)(x1)
    # x1 = BatchNormalization()(x1)
    feat_x = Dense(128, activation="linear")(x1)
    feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)

    model_top = Model(inputs=[img_input], outputs=feat_x)

    model_top.summary()

    """# model_3"""
    im_in1 = Input(shape=shape)
    im_in2 = Input(shape=shape)

    feat_x1 = model_top(im_in1)
    feat_x2 = model_top(im_in2)

    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

    model_final = Model(inputs=[im_in1, im_in2], outputs=lambda_merge)

    model_final.summary()

    adam = Adam(lr=0.001)

    sgd = SGD(lr=0.001, momentum=0.9)

    model_final.compile(optimizer=adam, loss=contrastive_loss)

    return model_final


def main():
    genx, geny = generator(2)
    val_gen = generator(2)
    model_final = create_network()
    print("------train start------")
    outputs = model_final.fit(genx, geny, validation_data=val_gen, steps_per_epoch=1, epochs=1,
                              validation_steps=2)
    print("------train finished------")
    """# Some model tests."""

    cop = create_similar_couple(path)
    diff = model_final.evaluate(
        [cop[0].reshape((1, shape[0], shape[1], shape[2])), cop[1].reshape((1, shape[0], shape[1], shape[2]))],
        np.array([0.]))
    print(diff)

    cop = create_wrong_couple(path)
    dist = model_final.predict(
        [cop[0].reshape((1, shape[0], shape[1], shape[2])), cop[1].reshape((1, shape[0], shape[1], shape[2]))])
    print(dist)

    """# Saving and loading the model.
    The next cells show both how to save the model weights and upload them into your Drive, and then how to retrieve those weights from the Drive to load a pre-trained model.
    """
    from keras import models

    def freeze(model):
        """Freeze model weights in every layer."""
        for layer in model.layers:
            layer.trainable = False

            if isinstance(layer, models.Model):
                freeze(layer)

    freeze(model_final)
    model_final.save_weights("weight_of_siamese_model.h5")
    # model_final.save("siamese_model.h5")


if __name__ == "__main__":
    main()

    # model_final = create_network()
    # model_final = model_final.load_weights('weight_of_siamese_model.h5')
    #
    # cop = create_wrong_couple(path)
    # dist = model_final.predict(
    #     [cop[0].reshape((1, shape[0], shape[1], shape[2])), cop[1].reshape((1, shape[0], shape[1], shape[2]))])
    # print(dist)
    #
    # cop = create_similar_couple(path)
    # diff = model_final.evaluate(
    #     [cop[0].reshape((1, shape[0], shape[1], shape[2])), cop[1].reshape((1, shape[0], shape[1], shape[2]))],
    #     np.array([0.]))
    # print(diff)

