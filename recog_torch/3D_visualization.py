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


def main(similar=False):
    obj = dxchange.reader.read_tiff(fname='./NUS 3D 数据/sample1/sample1 zscan 1xzoom 50um fiber 70um 1um step.tif',
                                    slc=None)  # Generate an object

    ang = tomopy.angles(180)  # Generate uniformly spaced tilt angles

    obj = obj.astype(np.float32)

    if similar:
        obj += np.random.uniform(size=obj.shape, low=-0.2, high=0.2)

    # test
    sim = tomopy.project(obj, ang)  # Calculate projections
    rec = tomopy.recon(sim, ang, algorithm='art')  # Reconstruct object
    print(rec.shape)

    # show 64th slice of the reconstructed object.
    mlab.imshow(rec[64], colormap='gray')
    mlab.show()

    return rec


if __name__ == "__main__":
    scalars = main()

    obj = mlab.contour3d(scalars, colormap='gray', transparent=True)   # transparent: 该对象可以透明表示，可以查看内部
    mlab.show()
