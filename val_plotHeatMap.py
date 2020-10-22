import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

sns.set(font_scale=0.5)


def draw_heat_map(df, rx_tick, sz_tick, sz_tick_num, rx_tick_num, x_label, z_label, map_title):
    # 用于画图
    ax = sns.heatmap(df, square=True, xticklabels=rx_tick, yticklabels=sz_tick, cmap="YlGnBu", vmax=20, annot=True,
                     fmt='.1f', linewidths=0.3)

    # ax.set_xticks(rx_tick_num)
    # ax.set_yticks(sz_tick_num)
    ax.set_xlabel(x_label)
    ax.set_ylabel(z_label)
    ax.set_title(map_title)
    save_path = "/home/aistudio/figure_and_csv/heatMap-" + map_title + '.png'
    plt.savefig(save_path, dpi=300)

    plt.axis([0, len(index), len(index), 0])

    plt.show()
    plt.close()


# main
path = './figure_and_csv/val_Transform_3D.csv'
data = pd.read_csv(path)
data['path_sample1'] = data['path_sample1'].apply(lambda x: x.split('/')[-1][:-4])
data['path_sample2'] = data['path_sample2'].apply(lambda x: x.split('/')[-1][:-4])
data.sort_values(by=['path_sample1', 'path_sample2'], inplace=True)
index = list(data['path_sample1'].drop_duplicates(keep="first"))
print(index)

columns = ["unnamed" + str(i) for i in range(len(index))]
heatMapData = pd.DataFrame(columns=columns)

loc = 1
for i in index:
    row = []
    col = index[:loc]
    # print(col)

    for c in col:
        v = data[(data.path_sample1 == i) & (data.path_sample2 == c) | (data.path_sample1 == c) & (
                    data.path_sample2 == i)].distance
        row.append(v.values[0])
    # print(row)
    while len(row) < len(index):
        row = row + [np.NaN]
    row = dict(zip(columns, row))
    # print(row)
    heatMapData = heatMapData.append(row, ignore_index=True)
    loc += 1

# print(heatMapData)

f, ax = plt.subplots(figsize=(10, 10))
tick = list(map(str, index))
# tick = range(0, len(index), 1)
rx_tick = tick
sz_tick = tick

draw_heat_map(heatMapData, rx_tick, sz_tick, sz_tick, rx_tick, '', '', path.split('/')[-1][:-4])
