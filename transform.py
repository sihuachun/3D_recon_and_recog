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
        image_augs = [image]
        List = [-10, -5, 5, 10]
        for angle in List:
            seq = iaa.Sequential([iaa.Rotate(rotate=angle)])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Shift(self, image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[:2]
        size = min(rows, cols) // 10
        image_augs = [image]
        List = [(size, 0), (-size, 0), (0, size), (0, -size)]
        for delta_x, delta_y in List:
            # delta_x<0左移，delta_x>0右移
            # delta_y<0上移，delta_y>0下移
            seq = iaa.Sequential([iaa.Affine(translate_px={"x": delta_x, "y": delta_y})])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Contrast(self, image, **kwargs):
        assert len(image.shape) == 3
        image_augs = [image]
        List = [0.6, 0.8, 1.0, 1.2]
        for alpha in List:
            seq = iaa.Sequential([iaa.contrast.LinearContrast(alpha, per_channel=False)])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Stain(self, image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[:2]
        size = min(rows, cols) // 10
        image_augs = [image]
        List = [(0, 0), (0, cols // 2), (rows // 2, 0), (rows // 2, cols // 2)]
        for x, y in List:
            image_aug = image.copy()
            image_aug[x:x + size, y:y + size, :] = 0
            image_augs.append(image_aug)
        return image_augs, List

    def Tilt(self, image, **kwargs):
        # 透射改变视角, 即倾斜图片
        assert len(image.shape) == 3
        image_augs = [image]
        List = [0.01, 0.03, 0.05, 0.07]
        for scale in List:
            seq = iaa.Sequential([iaa.PerspectiveTransform(scale)])
            image_aug = seq.augment_image(image)
            image_augs.append(image_aug)
        return image_augs, List

    def transform_2D(self):
        root_path = '2D'
        root_save_path = 'Transform_2D'
        file_path = os.listdir(root_path)
        function = [lambda x: self.Rotation(x), lambda x, rows=None, cols=None: self.Shift(x, rows=rows, cols=cols),
                    lambda x: self.Tilt(x), lambda x, rows=None, cols=None: self.Stain(x, rows, cols),
                    lambda x: self.Contrast(x)]
        class_name = ["Rotation", "Shift", "Tilt", "Stain", "Contrast"]
        for i in range(len(file_path)):
            f = function[i % len(function)]
            name = class_name[i % len(class_name)]
            file_name = file_path[i]
            path = os.path.join(root_path, file_name)
            image = cv2.imread(path)  # HWC
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
            height, width, _ = img.shape
            img = img[70:height - 70, 75:width - 50, :]
            image_augs, params = f(img)
            os.makedirs(os.path.join(root_save_path, name), exist_ok=True)
            cv2.imwrite(os.path.join(root_save_path, name, "{}_ori.jpg".format(file_name[:-4])), image_augs[0])
            for i in range(len(image_augs) - 1):
                image_aug = image_augs[i + 1]
                param = "{}_{}.jpg".format(file_name[:-4], params[i])
                save_path = os.path.join(root_save_path, name, param)
                cv2.imwrite(save_path, image_aug)


class Transform_3D:
    def Rotation(self, image, **kwargs):
        assert len(image.shape) == 3
        image_augs = [image]
        List = [-10, -5, 5, 10]
        for angle in List:
            seq = iaa.Sequential([iaa.Rotate(rotate=angle)])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Shift(self, image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[:2]
        size = min(rows, cols) // 10
        image_augs = [image]
        List = [(size, 0), (-size, 0), (0, size), (0, -size)]
        for delta_x, delta_y in List:
            # delta_x<0左移，delta_x>0右移
            # delta_y<0上移，delta_y>0下移
            seq = iaa.Sequential([iaa.Affine(translate_px={"x": delta_x, "y": delta_y})])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Contrast(self, image, **kwargs):
        assert len(image.shape) == 3
        image_augs = [image]
        List = [0.6, 0.8, 1.0, 1.2]
        for alpha in List:
            seq = iaa.Sequential([iaa.contrast.LinearContrast(alpha, per_channel=False)])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def Stain(self, image, rows=None, cols=None):
        assert len(image.shape) == 3
        if not rows or not cols:
            rows, cols = image.shape[:2]
        size = min(rows, cols) // 10
        image_augs = [image]
        List = [(0, 0), (0, cols // 2), (rows // 2, 0), (rows // 2, cols // 2)]
        for x, y in List:
            image_aug = image.copy()
            image_aug[:, x:x + size, y:y + size] = 0
            image_augs.append(image_aug)
        return image_augs, List

    def Tilt(self, image, **kwargs):
        # 透射改变视角, 即倾斜图片
        assert len(image.shape) == 3
        image_augs = [image]
        List = [0.01, 0.03, 0.05, 0.07]
        for scale in List:
            seq = iaa.Sequential([iaa.PerspectiveTransform(scale)])
            image_aug = seq.augment_images(image)
            image_augs.append(image_aug)
        return image_augs, List

    def transform_3D(self):
        root_path = 'sample2_recon'
        root_save_path = 'Transform_3D'
        file_path = os.listdir(root_path)
        function = [lambda x, rows, cols: self.Rotation(x, rows=rows, cols=cols),
                    lambda x, rows, cols: self.Shift(x, rows=rows, cols=cols),
                    lambda x, rows, cols: self.Tilt(x, rows=rows, cols=cols),
                    lambda x, rows, cols: self.Stain(x, rows=rows, cols=cols),
                    lambda x, rows, cols: self.Contrast(x, rows=rows, cols=cols)]
        class_name = ["Rotation", "Shift", "Tilt", "Stain", "Contrast"]
        for i in range(len(file_path)):
            f = function[i % len(function)]
            name = class_name[i % len(class_name)]
            file_name = file_path[i]
            path = os.path.join(root_path, file_name)
            scalar = tifffile.imread(path)
            scalar = (scalar * 255).astype(np.uint8)
            os.makedirs(os.path.join(root_save_path, name), exist_ok=True)
            rows, cols = scalar.shape[1:]
            scalars, params = f(scalar, rows=rows, cols=cols)
            file_name = file_name.split("_")[-1]
            tifffile.imwrite(os.path.join(root_save_path, name, "{}_ori.tif".format(file_name[:-4])), scalars[0])
            for i in range(len(scalars) - 1):
                image_aug = scalars[i + 1]
                param = "{}_{}.tif".format(file_name[:-4], params[i])
                save_path = os.path.join(root_save_path, name, param)
                tifffile.imwrite(save_path, image_aug)


if __name__ == "__main__":
    Transform_2D().transform_2D()
    Transform_3D().transform_3D()