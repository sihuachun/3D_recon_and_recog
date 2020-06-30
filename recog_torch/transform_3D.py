"""
self create similar image, then manual divide training set an testing set
"""



import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

readpath = "./NUS 3D 数据/sample1_standard"
writepath = "./sample1_standard_transform"


def Contrast_and_Brightness(alpha, beta, img):
    """改变对比度和亮度"""
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst


rlist = []
for dir, folder, file in os.walk(readpath):
    for i in file:
        t = "%s/%s" % (dir, i)
        rlist.append(t)
print(rlist)

# cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
for p in rlist:
    print(p)
    s_num = p.split('_')[-1][:-4]
    os.makedirs(writepath + "/s" + s_num)

    Object = tifffile.imread(p)
    wpath = writepath + "/s{}/A_step_{}.tif".format(s_num, s_num)
    tifffile.imwrite(wpath, Object)

    # spin
    spin_object = Object.copy()
    tilt_object = Object.copy()
    contrast_object = Object.copy()
    for cha in range(spin_object.shape[0]):
        img = spin_object[cha]
        rows, cols = img.shape
        # print(img[0])
        # spin image
        M = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=90, scale=1)
        spin = cv2.warpAffine(img, M, (cols, rows))
        spin_object[cha] = spin
        # print(spin.shape)
        # 透射改变视角, 即倾斜图片
        img = tilt_object[cha]
        pts1 = np.float32([[100, 50], [460, 50], [100, 360], [460, 360]])
        pts2 = np.float32([[50, 50], [430, 50], [50, 360], [430, 360]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        tilt = cv2.warpPerspective(img, M, (cols, rows))
        tilt_object[cha] = tilt
        # print(tilt[0])
        # 改变对比度和亮度
        img = contrast_object[cha]
        contrast = Contrast_and_Brightness(1.3, 1, img)
        contrast_object[cha] = contrast
        # print(contrast[0])
        # plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('origin')
        # plt.subplot(222), plt.imshow(spin, cmap='gray'), plt.title('spin')
        # plt.subplot(223), plt.imshow(tilt, cmap='gray'), plt.title('tilt')
        # plt.subplot(224), plt.imshow(contrast, cmap='gray'), plt.title('contrast')
        # plt.show()

    wpath = writepath + "/s{}/spin_{}.tif".format(s_num, 90)
    tifffile.imwrite(wpath, spin_object)
    wpath = writepath + "/s{}/tilt.tif".format(s_num)
    tifffile.imwrite(wpath, tilt_object)
    wpath = writepath + "/s{}/contrast.tif".format(s_num)
    tifffile.imwrite(wpath, contrast_object)

    # break

# cv2.destroyAllWindows()
