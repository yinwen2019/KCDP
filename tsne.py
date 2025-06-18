import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap
# E2S TGCA 1k
# Emo_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_TGCA_emo_v_1k.csv')
# Emo_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_TGCA_emo_l_1k.csv')
# SER_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_TGCA_ser_v_1k.csv')
# SER_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_TGCA_ser_l_1k.csv')
# E2S KADAP 1k
# Emo_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_emo_v_1k.csv')
# Emo_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_emo_l_1k.csv')
# SER_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_ser_v_1k.csv')
# SER_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_ser_l_1k.csv')
# E2S KADAP 4k
# Emo_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_emo_v_4k.csv')
# Emo_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_emo_l_4k.csv')
# SER_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_ser_v_4k.csv')
# SER_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_ser_l_4k.csv')
# S2E TGCA 1k
# Emo_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_TGCA_emo_v_1k.csv')
# Emo_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_TGCA_emo_l_1k.csv')
# SER_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_TGCA_ser_v_1k.csv')
# SER_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_TGCA_ser_l_1k.csv')
# S2E CLIEA 1k
# Emo_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_CLIEA_emo_v_1k.csv')
# Emo_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_CLIEA_emo_l_1k.csv')
# SER_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_CLIEA_ser_v_1k.csv')
# SER_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/S2E_CLIEA_ser_l_1k.csv')
# # E2S KADAP 100
Emo_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_emo_v_100.csv')
Emo_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_emo_l_100.csv')
SER_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_ser_v_100.csv')
SER_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_KADAP_ser_l_100.csv')
# E2S TGCA 100
# Emo_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_TGCA_emo_v_100.csv')
# Emo_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_TGCA_emo_l_100.csv')
# SER_v = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_TGCA_ser_v_100.csv')
# SER_l = pd.read_csv('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_TGCA_ser_l_100.csv')
# 将 pandas DataFrame 转换为 numpy 数组
# size = 1000
# ev = Emo_v.values[:size]
# el = Emo_l.values.flatten()[:size]
# sv = SER_v.values[:size]
# sl = SER_l.values.flatten()[:size]

df = pd.DataFrame(Emo_v)
df['Label'] = Emo_l
# 设置每个类别取样的数量
sample_size_e = min(df['Label'].value_counts())  # 可以取每个类别最小的样本数，或者指定数量
sampled_df = df.groupby('Label').apply(lambda x: x.sample(n=sample_size_e, random_state=42)).reset_index(drop=True)

# 提取结果
ev = sampled_df.drop(columns=['Label']).values  # 去掉标签列，保留样本
el = sampled_df['Label'].values  # 提取标签列

df = pd.DataFrame(SER_v)
df['Label'] = SER_l
# 设置每个类别取样的数量
sample_size_s = min(df['Label'].value_counts())  # 可以取每个类别最小的样本数，或者指定数量
sampled_df = df.groupby('Label').apply(lambda x: x.sample(n=sample_size_s, random_state=42)).reset_index(drop=True)

# 提取结果
sv = sampled_df.drop(columns=['Label']).values  # 去掉标签列，保留样本
sl = sampled_df['Label'].values  # 提取标签列

#tsne = TSNE(n_components=2, random_state=1120, perplexity=25, early_exaggeration=12, n_iter=900)
tsne = TSNE(n_components=2, random_state=1120, perplexity=20, early_exaggeration=10, n_iter=1000)
ev_tsne = tsne.fit_transform(ev)
sv_tsne = tsne.fit_transform(sv)

# umap_model = umap.UMAP(n_components=2, random_state=1120)
# ev_tsne = umap_model.fit_transform(ev)
# sv_tsne = umap_model.fit_transform(sv)
# 创建散点图
plt.figure(figsize=(10, 10))
# sns.set_theme(style="darkgrid")

colors = ['#3A6EA5', '#D94A4A']
# for domain_idx, (tsne_data, labels) in enumerate([(ev_tsne, el), (sv_tsne, sl)]):
#     for class_label in range(6):
#         mask = labels == class_label
#         # plt.scatter(
#         #     tsne_data[mask, 0],
#         #     tsne_data[mask, 1],
#         #     c=colors[domain_idx],
#         #     marker=markers[class_label],
#         #     s=10,  # 增大点的大小
#         #     edgecolor='white',  # 添加白色边缘
#         #     linewidth=1,
#         #     label=f'{domain_names[domain_idx]} - Class {class_label}'
#         # )
#         sns.scatterplot(
#             x=tsne_data[mask, 0],
#             y=tsne_data[mask, 1],
#             c=colors[domain_idx],
#             # marker=markers[class_label],
#             s=100,  # 增大点的大小
#             edgecolor='none',  # 添加白色边缘
#             label=f'{domain_names[domain_idx]} - Class {class_label}'
#         )
data = np.vstack([ev_tsne, sv_tsne])
labels = np.concatenate([el, sl])
domain = ["Real-world"] * sample_size_e * 6 + ["Stickers"] * sample_size_s* 6
print('domain e:'+ str(sample_size_e))
print('domain s:'+ str(sample_size_s))
df = pd.DataFrame(data, columns=["X", "Y"])
df["Domain"] = domain
df["Label"] = labels
#joint_plot=sns.jointplot(data=df, x="X", y="Y", hue="Domain", edgecolor='none', palette=colors)
#g.set_axis_labels("", "", fontsize=0)
g = sns.scatterplot(data=df,x='X', y='Y', hue='Domain', s=150, palette='Set1',edgecolor='none')
#g = sns.scatterplot(data=df, x='X', y='Y', hue='Domain', style='Label', s=150, palette='Set1',edgecolor='none', markers=['o', 'X', 's', 'D', '^', 'v'])


plt.xticks([])
plt.yticks([])


# 优化图例
# plt.legend(bbox_to_anchor=(1.05, 1),
#           loc='upper left',
#           borderaxespad=0.,
#           frameon=True,
#           fontsize=10)
# plt.gca().get_legend().remove()
plt.legend(title='Domain', loc='upper left', title_fontsize='24', fontsize='20')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/user/xxx/MultiModal/TGCA_PVT/tsne/E2S_kadap_100.pdf', dpi=300, bbox_inches='tight')
plt.show()