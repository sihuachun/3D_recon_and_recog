"""
2D data transform, create flip, tilt, contrast
"""


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

readpath = "./data_2D"
writepath = "./data2D_transform"


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
    s_num = p.split('/')[-1][:-4]
    os.makedirs(writepath + "/s" + s_num)

    image = cv2.imread(p)  # 常见图像格式
    rows, cols = image.shape[:2]
    wpath = writepath + "/s{}/{}.jpg".format(s_num, s_num)
    cv2.imwrite(wpath, image)

    # cv2.imshow('input_image', image)
    # cv2.waitKey(0)
    # # flip image
    # for i in [1, 0, -1]:
    #     # # Flipped Horizontally 水平翻转
    #     # h_flip = cv2.flip(image, 1)
    #     # # Flipped Vertically 垂直翻转
    #     # v_flip = cv2.flip(image, 0)
    #     # # Flipped Horizontally & Vertically 水平垂直翻转
    #     # hv_flip = cv2.flip(image, -1)
    #
    #     flip = cv2.flip(image, i)
    #     wpath = writepath + "/s{}/flip_{}.jpg".format(s_num, i)
    #     cv2.imwrite(wpath, flip)
    #
    #     # cv2.imshow('input_image', flip)
    #     # cv2.waitKey(0)

    # spin image
    for angle in [60, 90, 180]:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        spin = cv2.warpAffine(image, M, (cols, rows))
        wpath = writepath + "/s{}/spin_{}.jpg".format(s_num, angle)
        cv2.imwrite(wpath, spin)

    # 透射改变视角, 即倾斜图片
    pts1 = np.float32([[100, 50], [460, 50], [100, 360], [460, 360]])
    pts2 = np.float32([[50, 50], [430, 50], [50, 360], [430, 360]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    tilt = cv2.warpPerspective(image, M, (cols, rows))
    wpath = writepath + "/s{}/tilt.jpg".format(s_num)
    cv2.imwrite(wpath, tilt)

    contrast = Contrast_and_Brightness(1.3, 3, image)
    wpath = writepath + "/s{}/contrast.jpg".format(s_num)
    cv2.imwrite(wpath, contrast)
    #
    # plt.subplot(221), plt.imshow(image), plt.title('origin')
    # plt.subplot(222), plt.imshow(spin), plt.title('spin')
    # plt.subplot(223), plt.imshow(tilt), plt.title('tilt')
    # plt.subplot(224), plt.imshow(constrast), plt.title('contrast')
    # plt.show()
    # break

# cv2.destroyAllWindows()
