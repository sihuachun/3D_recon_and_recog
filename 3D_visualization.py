#!/usr/bin/env python 3.6.7
# -*- coding: utf-8 -*-
# author  : sihuachun
# date    : 2020/6/11
# software: PyCharm

"""
3D visualization
"""
import tomopy
import dxchange
import numpy as np
from mayavi import mlab
import cv2
import tifffile
import os


# The 'colormap' trait of an IsoSurfaceFactory instance must be 'Accent' or 'Blues' or 'BrBG' or 'BuGn' or 'BuPu' or
# 'CMRmap' or 'Dark2' or 'GnBu' or 'Greens' or 'Greys' or 'OrRd' or 'Oranges' or 'PRGn' or 'Paired' or 'Pastel1' or
# 'Pastel2' or 'PiYG' or 'PuBu' or 'PuBuGn' or 'PuOr' or 'PuRd' or 'Purples' or 'RdBu' or 'RdGy' or 'RdPu' or
# 'RdYlBu' or 'RdYlGn' or 'Reds' or 'Set1' or 'Set2' or 'Set3' or 'Spectral' or 'Vega10' or 'Vega20' or 'Vega20b' or
# 'Vega20c' or 'Wistia' or 'YlGn' or 'YlGnBu' or 'YlOrBr' or 'YlOrRd' or 'afmhot' or 'autumn' or 'binary' or
# 'black-white' or 'blue-red' or 'bone' or 'brg' or 'bwr' or 'cool' or 'coolwarm' or 'copper' or 'cubehelix' or
# 'file' or 'flag' or 'gist_earth' or 'gist_gray' or 'gist_heat' or 'gist_ncar' or 'gist_rainbow' or 'gist_stern' or
# 'gist_yarg' or 'gnuplot' or 'gnuplot2' or 'gray' or 'hot' or 'hsv' or 'inferno' or 'jet' or 'magma' or
# 'nipy_spectral' or 'ocean' or 'pink' or 'plasma' or 'prism' or 'rainbow' or 'seismic' or 'spectral' or 'spring' or
# 'summer' or 'terrain' or 'viridis' or 'winter', but a value of 'red' <class 'str'> was specified.


def main():
    obj = dxchange.reader.read_tiff(fname='./NUS 3D/sample1/sample1 zscan 1xzoom 50um fiber 70um 1um step.tif',
                                    slc=None)  # Generate an object

    ang = tomopy.angles(180)  # Generate uniformly spaced tilt angles
    obj = obj.astype(np.float32)
    sim = tomopy.project(obj, ang)  # Calculate projections
    rec = tomopy.recon(sim, ang, algorithm='art')  # Reconstruct object
    print(rec.shape)

    # show 64th slice of the reconstructed object.
    # mlab.imshow(rec[64], colormap='gray')
    # mlab.show()

    mlab.contour3d(rec, contours=1, colormap='gist_gray', transparent=True)  # transparent: 该对象可以透明表示，可以查看内部
    mlab.show()
    return


def structure_3D():
    import numpy as np
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d.axes3d import Axes3D
    # # 此处fig是二维
    # fig = plt.figure()
    # # 将二维转化为三维
    # axes3d = Axes3D(fig)
    # axes3d.contourf()
    scalars = tifffile.imread('./NUS 3D/sample1_standard/70um 1um step_1.tif')
    shape = scalars.shape
    print(shape)
    mid_scalars = scalars[shape[0] // 2]
    data = np.array(mid_scalars)
    # data = data[int(0.6 * shape[1]):int(0.8 * shape[1])]
    data = data[int(0.4 * shape[1]):int(0.6 * shape[1])]

    Z = []
    for row in data:
        # Z.append(row[int(0.6 * shape[1]):int(0.8 * shape[1])])
        Z.append(row[int(0.4 * shape[1]):int(0.6 * shape[1])])
    zz = np.array(Z)
    shape = zz.shape
    # 构造需要显示的值
    X = np.arange(0, shape[0], step=1)  # X轴的坐标
    Y = np.arange(0, shape[1], step=1)  # Y轴的坐标

    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    bottom = np.zeros_like(X)  # 设置柱状图的底端位值
    Z = zz.ravel()  # 扁平化矩阵
    width = height = 2  # 每一个柱子的长和宽

    # mlab.barchart(X, Y, Z, colormap='hot')
    # mlab.show()

    # # 绘图设置
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # 三维坐标轴
    ax.bar3d(X, Y, bottom, width, height, Z, shade=True, color='red')

    cs = ax.contourf(xx, yy, zz, 60, cmap='hot')
    cbar = fig.colorbar(cs)

    # 坐标轴设置
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z(value)')
    plt.savefig("./heatmap_jpg/structure1_3D_1.jpg")
    plt.show()
    return


def view(path='recon_data/Transform_3D/Contrast/8_ori.tif'):
    image = tifffile.imread(path)
    obj = image.astype(np.float32)
    print(obj.shape, type(obj))
    obj = mlab.contour3d(obj, contours=1, colormap='gray', transparent=True)  # transparent: 该对象可以透明表示，可以查看内部
    mlab.show()


def recon(path='NUS 3D/sample2', save_path='./model/'):
    save_path = os.path.join(save_path, path.split('/')[-1]+'_recon')
    os.makedirs(save_path, exist_ok=True)
    paths = os.listdir(path)
    paths = sorted(paths)
    for file_name in paths[:5]:
        ph = os.path.join(path, file_name)
        sh = os.path.join(save_path, file_name)
        obj = dxchange.reader.read_tiff(fname=ph, slc=None)  # Generate an object
        ang = tomopy.angles(180)  # Generate uniformly spaced tilt angles
        obj = obj.astype(np.float32)
        sim = tomopy.project(obj, ang)  # Calculate projections
        rec = tomopy.recon(sim, ang, algorithm='art')  # Reconstruct object
        tifffile.imwrite(sh, rec)


if __name__ == "__main__":
    # structure_3D()
    # view()
    main()
    # recon()