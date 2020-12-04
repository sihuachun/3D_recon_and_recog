import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# seabornDF =pd.read_excel('raw_data_of_violinplot_ORE.xlsx', index_col=False)
seabornDF1 = pd.read_csv('./figure_and_csv/3D_3D.csv', index_col=False)
seabornDF1['Class'] = "3D"
seabornDF2 = pd.read_csv('./figure_and_csv/2D_2D.csv', index_col=False)
seabornDF2['Class'] = '2D'
seabornDF = seabornDF1.append(seabornDF2, ignore_index=True)
seabornDF['label'] = seabornDF['label'].apply(lambda x: 'same' if x == 0 else 'different')
print(seabornDF.iloc[0, :])

# need
sns.stripplot(x="label", y="distance", data=seabornDF,
              hue="Class",
              dodge=True,
              jitter=0.3,
              size=5, linewidth=1,
              order=['same', 'different'],
              zorder=10,  # 决定图层级别
              )

ax = sns.boxplot(x="label", y="distance",
                 hue="Class",
                 linewidth=3,
                 zorder=40,
                 whis=1.5,
                 # boxprops=dict(alpha=.1),
                 order=['same', 'different'],
                 data=seabornDF)

for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .1))
# sns.despine(offset=10, trim=True)

handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2])  # 只显示前5个label

# sns.despine(left=True)
# sns.spines['left'].set_color('b')
plt.savefig("./figure_and_csv/wide_form_violinplot.png")
plt.show()
