#!/usr/bin/env python 3.6.7
# -*- coding: utf-8 -*-
# author  : sihuachun
# date    : 2020/10/23
# software: PyCharm


import random
import os
import tifffile
import cv2
from imgaug import augmenters as iaa


def make_3D_dataset():
    folders = ["../NUS 3D/sample1", "../NUS 3D/sample1_standard"]
    paths = []
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            paths.append(os.path.join(folder, file))
    print("all dataset length: {}".format(len(paths)))
    random.shuffle(paths)
    dilation = int(len(paths) * 0.8)
    train_path = paths[:dilation]
    test_path = paths[dilation:]
    print("train dataset length: {}".format(len(train_path)))
    print("test dataset length: {}".format(len(test_path)))
    target_path = "./3D"
    step = 1
    os.makedirs(os.path.join(target_path, "train"), exist_ok=True)
    for path in train_path:
        image = tifffile.imread(path)
        tifffile.imwrite(os.path.join(target_path, "train/{}.tif".format(step)), image)
        step += 1
    os.makedirs(os.path.join(target_path, "test"), exist_ok=True)
    for path in test_path:
        image = tifffile.imread(path)
        tifffile.imwrite(os.path.join(target_path, "test/{}.tif".format(step)), image)
        step += 1


def make_2D_dataset():
    folders = ["../NUS 3D/data2D_200910"]
    paths = []
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            paths.append(os.path.join(folder, file))
    paths = paths[:50]
    print("all dataset length: {}".format(len(paths)))
    random.shuffle(paths)
    dilation = int(len(paths) * 0.8)
    train_path = paths[:dilation]
    test_path = paths[dilation:]
    print("train dataset length: {}".format(len(train_path)))
    print("test dataset length: {}".format(len(test_path)))
    target_path = "./2D"
    step = 1
    os.makedirs(os.path.join(target_path, "train"), exist_ok=True)
    for path in train_path:
        image = cv2.imread(path)
        cv2.imwrite(os.path.join(target_path, "train/{}.png".format(step)), image)
        step += 1
    os.makedirs(os.path.join(target_path, "test"), exist_ok=True)
    for path in test_path:
        image = cv2.imread(path)
        cv2.imwrite(os.path.join(target_path, "test/{}.png".format(step)), image)
        step += 1


def make_3D_same_dataset():
    def Rotation(image, **kwargs):
        assert len(image.shape) == 3
        image_augs = []
        List = [0, -2, 2, -4, 4]  # Rotation angle
        for angle in List:
            seq = iaa.Sequential([iaa.Rotate(rotate=angle)])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Shift(image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[1:]
        size = min(rows, cols) // 20
        image_augs = []
        List = [(0, 0), (int(1 * size), 0), (-int(1 * size), 0), (0, int(1 * size)), (0, -int(1 * size))]
        for delta_x, delta_y in List:
            # delta_x<0左移，delta_x>0右移
            # delta_y<0上移，delta_y>0下移
            seq = iaa.Sequential([iaa.Affine(translate_px={"x": delta_x, "y": delta_y})])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Stain(image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[1:]
        size = min(rows, cols) // 40
        image_augs = []
        List = [(rows // 4, cols // 4), (rows // 4, 3 * cols // 4), (2 * rows // 4, 2 * cols // 4),
                (3 * rows // 4, 3 * cols // 4), (3 * rows // 4, cols // 4)]
        for x, y in List:
            image_aug = image.copy()
            image_aug[:, x - size:x + size, y - size:y + size] = 0
            image_augs.append(image_aug)
        return image_augs, List

    def Tilt(image, **kwargs):
        # 透射改变视角, 即倾斜图片
        assert len(image.shape) == 3
        image_augs = []
        List = [0, 0.005, 0.01, 0.015, 0.02]  # 倾斜程度
        for scale in List:
            seq = iaa.Sequential([iaa.PerspectiveTransform(scale)])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Contrast(image, **kwargs):
        assert len(image.shape) == 3
        image_augs = []
        List = [0.8, 0.9, 1.0, 1.1, 1.2]  # 对比度比例
        for alpha in List:
            seq = iaa.Sequential([iaa.contrast.LinearContrast(alpha, per_channel=False)])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    root_path = '3D/train'
    root_save_path = '3D/same'
    file_path = os.listdir(root_path)[:5]
    function = [lambda x: Rotation(x),
                lambda x, rows=None, cols=None: Shift(x, rows=rows, cols=cols),
                lambda x: Tilt(x),
                lambda x, rows=None, cols=None: Stain(x, rows=rows, cols=cols),
                lambda x: Contrast(x)]
    class_name = ["Rotation", "Shift", "Tilt", "Stain", "Contrast"]
    for i in range(len(file_path)):
        f = function[i % len(function)]
        name = class_name[i % len(class_name)]
        file_name = file_path[i]
        path = os.path.join(root_path, file_name)
        scalar = tifffile.imread(path)
        # print(scalar.shape)
        # print(scalar.dtype)
        os.makedirs(os.path.join(root_save_path, name), exist_ok=True)
        scalars, ratios = f(scalar)
        for i in range(len(scalars)):
            image_aug = scalars[i]
            print(image_aug.shape)
            print(image_aug.dtype)
            # return
            f_name = "{}_{}.tif".format(file_name[:-4], ratios[i])
            save_path = os.path.join(root_save_path, name, f_name)
            tifffile.imwrite(save_path, image_aug)


def make_2D_same_dataset():
    def Rotation(image, **kwargs):
        assert len(image.shape) == 3
        image_augs = []
        List = [0, -2, 2, -4, 4]  # Rotation angle
        for angle in List:
            seq = iaa.Sequential([iaa.Rotate(rotate=angle)])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Shift(image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[:2]
        size = min(rows, cols) // 20
        image_augs = []
        List = [(0, 0), (int(1 * size), 0), (-int(1 * size), 0), (0, int(1 * size)), (0, -int(1 * size))]
        for delta_x, delta_y in List:
            # delta_x<0左移，delta_x>0右移
            # delta_y<0上移，delta_y>0下移
            seq = iaa.Sequential([iaa.Affine(translate_px={"x": delta_x, "y": delta_y})])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Stain(image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[:2]
        size = min(rows, cols) // 40
        image_augs = []
        List = [(rows // 4, cols // 4), (rows // 4, 3 * cols // 4), (2 * rows // 4, 2 * cols // 4),
                (3 * rows // 4, 3 * cols // 4), (3 * rows // 4, cols // 4)]
        for x, y in List:
            image_aug = image.copy()
            image_aug[x - size:x + size, y - size:y + size, :] = 0
            image_augs.append(image_aug)
        return image_augs, List

    def Tilt(image, **kwargs):
        # 透射改变视角, 即倾斜图片
        assert len(image.shape) == 3
        image_augs = []
        List = [0, 0.005, 0.01, 0.015, 0.02]  # 倾斜程度
        for scale in List:
            seq = iaa.Sequential([iaa.PerspectiveTransform(scale)])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Contrast(image, **kwargs):
        assert len(image.shape) == 3
        image_augs = []
        List = [0.8, 0.9, 1.0, 1.1, 1.2]  # 对比度比例
        for alpha in List:
            seq = iaa.Sequential([iaa.contrast.LinearContrast(alpha, per_channel=False)])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    root_path = '2D/train'
    root_save_path = '2D/same'
    file_path = os.listdir(root_path)[:5]
    function = [lambda x: Rotation(x),
                lambda x, rows=None, cols=None: Shift(x, rows=rows, cols=cols),
                lambda x: Tilt(x),
                lambda x, rows=None, cols=None: Stain(x, rows=rows, cols=cols),
                lambda x: Contrast(x)]
    class_name = ["Rotation", "Shift", "Tilt", "Stain", "Contrast"]
    for i in range(len(file_path)):
        f = function[i % len(function)]
        name = class_name[i % len(class_name)]
        file_name = file_path[i]
        path = os.path.join(root_path, file_name)
        scalar = cv2.imread(path)
        os.makedirs(os.path.join(root_save_path, name), exist_ok=True)
        scalars, ratios = f(scalar)
        for i in range(len(scalars)):
            image_aug = scalars[i]
            # print(image_aug.dtype)
            f_name = "{}_{}.png".format(file_name[:-4], ratios[i])
            save_path = os.path.join(root_save_path, name, f_name)
            cv2.imwrite(save_path, image_aug)


if __name__ == "__main__":
    # make_3D_dataset()
    # make_2D_dataset()
    make_3D_same_dataset()
    # make_2D_same_dataset()
