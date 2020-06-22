"""
self create similar image, then manual divide training set an testing set
"""


import cv2
import os
import numpy as np


readpath = "./data_2D"
writepath = "./data2D"

rlist = []
for dir, folder, file in os.walk(readpath):
    for i in file:
        t = "%s/%s" % (dir, i)
        rlist.append(t)
print(rlist)

cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
for p in rlist:
    s_num = p.split('/')[-1][:-4]
    os.makedirs(writepath + "/s" + s_num)

    img = cv2.imread(p)  # 常见图像格式
    wpath = writepath + "/s{}/{}.jpg".format(s_num, s_num)
    cv2.imwrite(wpath, img)

    # cv2.imshow('input_image', img)
    # cv2.waitKey(0)

    for i in range(6):
        rand = np.random.uniform(low=0, high=0.002, size=(480, 640, 3))
        img_ = img + rand
        wpath = writepath + "/s{}/{}-{}.jpg".format(s_num, s_num, i)
        cv2.imwrite(wpath, img_)

        # cv2.imshow('input_image', img_)
        # cv2.waitKey(0)

cv2.destroyAllWindows()
