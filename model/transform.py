#!/usr/bin/env python 3.6.7
# -*- coding: utf-8 -*-
# author  : sihuachun
# date    : 2020/10/20
# software: PyCharm


from imgaug import augmenters as iaa  # 引入数据增强的包
import numpy as np
import cv2
import os
import tifffile

np.random.seed(23)


class Transform_2D:
    def Rotation(self, image, **kwargs):
        assert len(image.shape) == 3
        image_augs = []
        List = [0, 30, 50, 70, 90]  # Rotation angle
        for angle in List:
            seq = iaa.Sequential([iaa.Rotate(rotate=angle)])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Shift(self, image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[:2]
        size = min(rows, cols) // 20
        size = 25
        image_augs = []
        List = [(0, 0), (int(1 * size), 0), (2 * size, 0), (int(3 * size), 0), (4 * size, 0)]
        for delta_x, delta_y in List:
            # delta_x<0左移，delta_x>0右移
            # delta_y<0上移，delta_y>0下移
            seq = iaa.Sequential([iaa.Affine(translate_px={"x": delta_x, "y": delta_y})])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Contrast(self, image, **kwargs):
        assert len(image.shape) == 3
        image_augs = []
        List = [0.6, 0.8, 1.0, 1.2, 1.4]
        for alpha in List:
            seq = iaa.Sequential([iaa.contrast.LinearContrast(alpha, per_channel=False)])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Stain(self, image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[:2]
        size = min(rows, cols) // 20
        size = 25
        x, y = rows // 2, cols // 2
        image_augs = []
        List = [0 * size, 1 * size, 2 * size, 3 * size, 4 * size]
        for s in List:
            image_aug = image.copy()
            s_half = int(s / 2)
            image_aug[x - s_half:x + s_half, y - s_half:y + s_half, :] = 0
            image_augs.append(image_aug)
        return image_augs, List

    def Tilt(self, image, **kwargs):
        # 透射改变视角, 即倾斜图片
        assert len(image.shape) == 3
        image_augs = []
        List = [0, 0.03, 0.05, 0.07, 0.09]
        for scale in List:
            seq = iaa.Sequential([iaa.PerspectiveTransform(scale)])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def transform_2D(self):
        root_path = '2D/test'
        root_save_path = 'valDataset/2D'
        file_path = os.listdir(root_path)[:5]
        function = [lambda x: self.Rotation(x),
                    lambda x, rows=None, cols=None: self.Shift(x, rows=rows, cols=cols),
                    lambda x: self.Tilt(x),
                    lambda x, rows=None, cols=None: self.Stain(x, rows=rows, cols=cols),
                    lambda x: self.Contrast(x)]
        class_name = ["Rotation", "Shift", "Tilt", "Stain", "Contrast"]
        for i in range(len(file_path)):
            f = function[i % len(function)]
            name = class_name[i % len(class_name)]
            file_name = file_path[i]
            path = os.path.join(root_path, file_name)
            image = cv2.imread(path)  # HWC
            # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
            # height, width, _ = image.shape
            # image = image[70:height - 70, 75:width - 50, :]  # 修建图片，去除黑边
            image_augs, params = f(image)
            os.makedirs(os.path.join(root_save_path, name), exist_ok=True)
            for i in range(len(image_augs)):
                image_aug = image_augs[i]
                f_name = "{}_{}{}.png".format(file_name[:-4], name, params[i])
                save_path = os.path.join(root_save_path, name, f_name)
                # image_aug = cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, image_aug)


class Transform_3D:
    def Rotation(self, image, **kwargs):
        assert len(image.shape) == 3
        image_augs = []
        List = [0, 30, 50, 70, 90]  # Rotation angle
        for angle in List:
            seq = iaa.Sequential([iaa.Rotate(rotate=angle)])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Shift(self, image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[1:]
        size = min(rows, cols) // 20
        image_augs = []
        List = [(0, 0), (int(1 * size), 0), (2 * size, 0), (int(3 * size), 0), (4 * size, 0)]
        for delta_x, delta_y in List:
            # delta_x<0左移，delta_x>0右移
            # delta_y<0上移，delta_y>0下移
            seq = iaa.Sequential([iaa.Affine(translate_px={"x": delta_x, "y": delta_y})])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Contrast(self, image, **kwargs):
        assert len(image.shape) == 3
        image_augs = []
        List = [0.6, 0.8, 1.0, 1.2, 1.4]
        for alpha in List:
            seq = iaa.Sequential([iaa.contrast.LinearContrast(alpha, per_channel=False)])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Stain(self, image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[1:]
        size = min(rows, cols) // 20
        x, y = rows // 2, cols // 2
        image_augs = []
        List = [0 * size, 1 * size, 2 * size, 3 * size, 4 * size]
        for s in List:
            image_aug = image.copy()
            s_half = int(s / 2)
            image_aug[:, x - s_half:x + s_half, y - s_half:y + s_half] = 0
            image_augs.append(image_aug)
        return image_augs, List

    def Tilt(self, image, **kwargs):
        # 透射改变视角, 即倾斜图片
        assert len(image.shape) == 3
        image_augs = []
        List = [0, 0.03, 0.05, 0.07, 0.09]
        for scale in List:
            seq = iaa.Sequential([iaa.PerspectiveTransform(scale)])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def transform_3D(self):
        root_path = '3D/test'
        root_save_path = 'valDataset/3D'
        file_path = os.listdir(root_path)[5:10]
        function = [lambda x: self.Rotation(x),
                    lambda x, rows=None, cols=None: self.Shift(x, rows=rows, cols=cols),
                    lambda x: self.Tilt(x),
                    lambda x, rows=None, cols=None: self.Stain(x, rows=rows, cols=cols),
                    lambda x: self.Contrast(x)]
        class_name = ["Rotation", "Shift", "Tilt", "Stain", "Contrast"]
        for i in range(len(file_path)):
            f = function[i % len(function)]
            name = class_name[i % len(class_name)]
            file_name = file_path[i]
            path = os.path.join(root_path, file_name)
            scalar = tifffile.imread(path)
            # print(scalar.dtype)
            # scalar = (scalar * 255).astype(np.uint8)
            os.makedirs(os.path.join(root_save_path, name), exist_ok=True)
            scalars, params = f(scalar)
            for i in range(len(scalars)):
                image_aug = scalars[i]
                f_name = "{}_{}{}.tif".format(file_name[:-4], name, params[i])
                save_path = os.path.join(root_save_path, name, f_name)
                tifffile.imwrite(save_path, image_aug)


if __name__ == "__main__":
    Transform_2D().transform_2D()
    # Transform_3D().transform_3D()
