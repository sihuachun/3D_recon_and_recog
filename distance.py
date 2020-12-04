import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def dis_3D():
    data3D = pd.read_csv('./figure_and_csv/train_3D.csv')
    data3D['path_sample1'] = data3D['path_sample1'].apply(lambda x: x.split('_')[-1][:-4])
    data3D['path_sample2'] = data3D['path_sample2'].apply(lambda x: x.split('_')[-1][:-4])
    data3D.sort_values(by=['path_sample1', 'path_sample2'], inplace=True)
    dist = data3D['distance']
    dist = dist.loc[dist > 0]
    # dist.sort_values()
    print("3D_quantile Q1: {}".format(dist.quantile(0.25)))
    print("3D_quantile Q3: {}".format(dist.quantile(0.75)))
    return dist


def dis_2D():
    data2D = pd.read_csv('./figure_and_csv/train_2D.csv')
    data2D['path_sample1'] = data2D['path_sample1'].apply(lambda x: x.split('_')[-1][:-4])
    data2D['path_sample2'] = data2D['path_sample2'].apply(lambda x: x.split('_')[-1][:-4])
    data2D.sort_values(by=['path_sample1', 'path_sample2'], inplace=True)
    dist = data2D['distance']
    dist = dist.loc[dist > 0]
    # dist.sort_values()
    print("2D_quantile Q1: {}".format(dist.quantile(0.25)))
    print("2D_quantile Q3: {}".format(dist.quantile(0.75)))
    return dist


dis3D, dis2D = dis_3D(), dis_2D()
print("Max distance: dis_2D = {}, dis_3D = {}".format(max(dis2D), max(dis3D)))
print("Min distance: dis_2D = {}, dis_3D = {}".format(min(dis2D), min(dis3D)))
print("Average distance: ave_2D = {}, ave_3D = {}".format(dis2D.mean(), dis3D.mean()))
distance = dis3D - dis2D
print("Average distance: {}".format(distance.mean()))
print("Positive difference of distance: {}".format(distance[distance > 0].size))


def dis_puf(k):
    dataPUF = pd.read_csv('./figure_and_csv/kind{}_2D.csv'.format(k))
    dataPUF['path_sample1'] = dataPUF['path_sample1'].apply(lambda x: x.split('/')[-1][2:-4])
    dataPUF['path_sample2'] = dataPUF['path_sample2'].apply(lambda x: x.split('/')[-1][2:-4])
    dataPUF.sort_values(by=['path_sample1', 'path_sample2'], inplace=True)
    dist = dataPUF['distance']
    dist = dist.loc[dist > 0]
    return dist.mean(), min(dist)


# print("PUF: Mean, Min")
# mean_list = []
# min_list = []
# for k in range(1, 6):
#     _mean, _min = dis_puf(k)
#     mean_list.append(_mean)
#     min_list.append(_min)
# print(sum(mean_list) / len(mean_list), min(min_list))


def plot_heat_map_puf():
    sns.set(font_scale=0.5)

    def draw_heat_map(df, rx_tick, sz_tick, sz_tick_num, rx_tick_num, x_label, z_label, map_title):
        # 用于画图
        ax = sns.heatmap(df, square=True, xticklabels=rx_tick, yticklabels=sz_tick, cmap="YlGnBu", vmax=5000,
                         annot=True, fmt='.1f', linewidths=0.3)
        # ax.set_xticks(rx_tick_num)
        # ax.set_yticks(sz_tick_num)
        ax.set_xlabel(x_label)
        ax.set_ylabel(z_label)
        ax.set_title(map_title)
        save_path = "./figure_and_csv/heatMap-" + map_title + '.png'
        plt.savefig(save_path, dpi=300)
        plt.axis([0, len(index), len(index), 0])
        plt.show()
        plt.close()

    # main
    df = pd.DataFrame()
    for i in range(1, 6):
        print(i)
        path = './figure_and_csv/kind{}_2D.csv'.format(i)
        data = pd.read_csv(path)
        data["path_sample1"] = data["path_sample1"].apply(lambda x: "system{}-".format(i) + x.split('/')[-1][2:-4])
        data["path_sample2"] = data["path_sample2"].apply(lambda x: "system{}-".format(i) + x.split('/')[-1][2:-4])

        data["system".format(i)] = i
        df = df.append(data, ignore_index=True)
    print(df)
    df.sort_values(by=['path_sample1', 'path_sample2'], inplace=True)
    index = list(df['path_sample1'].drop_duplicates(keep="first"))
    print(index)

    columns = ["unnamed" + str(i) for i in range(len(index))]
    heatMapData = pd.DataFrame(columns=columns)
    loc = 1
    system = -1
    pre_NaN = 0
    for i in index:
        if i[:7] != system:
            system = i[:7]
            pre_NaN = loc - 1
        # print(system)
        row = [np.NaN for _ in range(pre_NaN)]
        col = index[pre_NaN:loc]
        for c in col:
            v = df[(df.path_sample1 == i) & (df.path_sample2 == c) | (df.path_sample1 == c) & (
                    df.path_sample2 == i)].distance
            row.append(v.values[0])
        # print(row)
        while len(row) < len(index):
            row.append(np.NaN)
        row = dict(zip(columns, row))
        heatMapData = heatMapData.append(row, ignore_index=True)
        loc += 1

    # print(heatMapData)

    f, ax = plt.subplots(figsize=(10, 10))
    # tick = range(0, len(index), 1)
    rx_tick = index
    sz_tick = index

    draw_heat_map(heatMapData, rx_tick, sz_tick, sz_tick, rx_tick, '', '', "PUF-System")


# plot_heat_map_puf()


def cacu_error_rate(k1=512, k2=485):
    df_3D = pd.read_csv("./figure_and_csv/3D_3D.csv")
    error_df_3D = df_3D.loc[df_3D['distance'] < k1]
    error_rate = len(error_df_3D) / len(df_3D)
    print("3D test dataset error rate {}".format(error_rate))
    df_2D = pd.read_csv("./figure_and_csv/2D_2D.csv")
    error_df_2D = df_2D.loc[df_2D['distance'] < k2]
    error_rate = len(error_df_2D) / len(df_2D)
    print("2D test data error rate {}".format(error_rate))


cacu_error_rate()
