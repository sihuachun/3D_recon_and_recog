# import numpy as np; np.random.seed(0)
# import seaborn as sns; sns.set()
# uniform_data = np.random.rand(10, 12)
# ax = sns.heatmap(uniform_data)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import pandas as pd

sns.set()
import numpy as np


def draw_heat_map(df, rx_tick, sz_tick, sz_tick_num, rx_tick_num, x_label, z_label, map_title):
    # 用于画图
    #c_map = sns.cubehelix_palette(start=1.6, light=0.8, as_cmap=True, reverse=True)
    #plt.subplots(figsize=(6, 6))
    ax = sns.heatmap(df, square=True, xticklabels=rx_tick, yticklabels=sz_tick, cmap="YlGnBu")

    ax.set_xticks(rx_tick_num)
    ax.set_yticks(sz_tick_num)

    ax.set_xlabel(x_label)
    ax.set_ylabel(z_label)
    ax.set_title(map_title)
    plt.savefig(map_title + '.png', dpi=300)

    plt.axis([0, 25, 25, 0])

    plt.show()
    plt.close()


#heatMapData = np.load("heatmapData_1222_296.npy")
heatMapData = pd.read_excel("HeatMapData.xlsx")
# np.savetxt('heatmapData_316.txt', heatMapData)

# [rows, cols] = heatMapData.shape
# print(rows, cols)
# for i in range(337):
#     for j in range(i):
#         #print(heatMapData[i, j])
#         if 1 == heatMapData[i, j] and i!=j:
#             heatMapData = np.delete(heatMapData, i, 0)  # 删除行
#             heatMapData = np.delete(heatMapData, i, 1)  # 删除列
#
# heatMapData[5, 0] = heatMapData[5, 2]
# heatMapData[0, 5] = heatMapData[5, 2]
#
# print(heatMapData.shape)
# #print(heatMapData)
#
# outlist = [52, 57, 72, 77, 92, 97,
#           112, 117, 132, 137, 152, 157, 172, 177, 192, 197, 212, 217, 242, 247]
#
# heatMapData = np.delete(heatMapData, outlist, 0)  # 删除行
# heatMapData = np.delete(heatMapData, outlist, 1)  # 删除行
#
# print(heatMapData.shape)
#
# np.save("heatmapData316.npy", heatMapData)

#f, ax = plt.subplots(figsize=(10, 9))
f, ax = plt.subplots()

#sns.heatmap(heatMapData, ax=ax, square=True)

rx_tick = range(0, 25, 5)
sz_tick = range(0, 25, 5)

draw_heat_map(heatMapData, rx_tick, sz_tick, sz_tick, rx_tick,'', '','HeatMap')

#ax = sns.heatmap(heatMapData)

#plt.savefig('heatmap.png')


#plt.show()

