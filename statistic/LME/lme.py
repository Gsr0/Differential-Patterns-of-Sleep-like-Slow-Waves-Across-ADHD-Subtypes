import pandas as pd
import numpy as np
import mne
from scipy import stats
import matplotlib.pyplot as plt

# --- 1. 参数设置与数据加载 ---
# 请将 'your_data_file.xlsx' 替换为您的Excel文件实际路径
file_path = '所有慢波参数分电极总表.xlsx'
# 请将 'Sheet1' 替换为您的Excel中的工作表名称
sheet_name = 'Sheet1'

# 您提供的电极名称列表 (共31个)
# 注意：您的描述是30个电极，但列表有31个。代码将以数据中的实际电极为准。
# 如果您的数据中channel列只到30，代码会自动截取前30个名称。
electrode_names = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
    'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5',
    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8','POz', 'O1','Oz', 'O2'
]

# 需要进行检验的参数列名
# 您可以修改或扩充这个列表来分析所有感兴趣的参数
params_to_test = [
    'maxnegpkamp', 'maxpospkamp', 'mxdnslp', 'mxupslp', 'sw_density', 'mean_duration'
]

# 组别定义
group1_id = 1  # ADHD 1型
group3_id = 3  # ADHD 3型

# 置换检验参数
p_threshold_cluster = 0.025  # 用于形成簇的p值阈值
p_threshold_monte_carlo = 0.05  # 用于判断簇是否显著的蒙特卡洛p值
n_permutations = 5000  # 置换次数，建议至少1000，5000更稳定

# 加载数据
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print("Excel文件加载成功。")
    print(f"数据包含 {df['SubjectID'].nunique()} 名被试。")
except FileNotFoundError:
    print(f"错误：未找到文件 '{file_path}'。请检查文件路径是否正确。")
    exit()

# --- 2. MNE准备工作：创建电极信息对象 ---
# 从数据中获取实际使用的电极数量和编号
ch_numbers_in_data = sorted(df['Channel'].unique())
n_channels = len(ch_numbers_in_data)
# 根据数据中的电极数量，从您提供的列表中获取对应的电极名称
ch_names = electrode_names[:n_channels]

print(f"\n数据中检测到 {n_channels} 个电极。")
print(f"将要使用的电极名称: {ch_names}")

# 创建MNE所需的info结构，它包含了电极的基本信息
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')  # sfreq是采样率，这里可以任意设置，因为我们分析的是非时序数据

# 设置电极位置。MNE会自动从其内置的标准10-20系统中寻找电极位置
# 如果您有.sfp文件，也可以使用 mne.channels.read_custom_montage('your_file.sfp') 来加载
montage = mne.channels.read_custom_montage('locations.sfp')
info.set_montage(montage, on_missing='warn')  # on_missing='warn' 会对找不到位置的电极发出警告

# --- 3. 定义电极邻近关系 ---
# 这是基于簇检验的关键一步，用于定义哪些电极是“相邻”的
# MNE可以根据电极的3D位置自动计算邻近关系
adjacency, adj_ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')

# 检查邻近关系矩阵中的电极顺序是否与info对象一致
if adj_ch_names != ch_names:
    print("警告：邻近矩阵的电极顺序与数据不匹配，正在尝试重新索引。")
    # 如果不匹配，需要重新索引邻近矩阵以确保顺序正确
    adj_indices = [adj_ch_names.index(ch) for ch in ch_names]
    adjacency = adjacency[np.ix_(adj_indices, adj_indices)]

print("\n电极邻近关系矩阵创建完成。")


# --- 4. 数据整理：将长格式数据转换为适合检验的宽格式 ---
# 原始数据是长格式（每个被试30行），需要转换为宽格式（每个被试1行，30个电极为列）
def prepare_data_for_test(df, param_name, group1, group2):
    """
    将数据从长格式整理为宽格式，并按组别分开。
    返回两个数组，分别是两个组的数据，形状为 (n_subjects, n_channels)。
    """
    # 筛选出当前要分析的参数和两个组别的数据
    df_param = df[df['Group'].isin([group1, group2])][['SubjectID', 'Group', 'Channel', param_name]]

    # 数据透视/转换
    df_pivot = df_param.pivot_table(index=['SubjectID', 'Group'], columns='Channel', values=param_name)

    # 分离两个组的数据
    data_g1 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group1].values
    data_g3 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group2].values

    return data_g1, data_g3


# --- 5. 循环执行置换检验并可视化 ---

# 将t检验的p值阈值转换为t值阈值
# 我们将进行双边检验，所以p值要除以2
t_threshold = stats.t.ppf(1.0 - p_threshold_cluster / 2, df=df['SubjectID'].nunique() - 2)
print(f"\n开始进行统计检验...")
print(f"用于形成簇的p值阈值: {p_threshold_cluster} (双边), 对应的t值阈值: {t_threshold:.3f}")
print(f"用于判断簇显著性的蒙特卡洛p值阈值: {p_threshold_monte_carlo}")
print("-" * 50)

# 遍历每一个需要检验的参数
for param in params_to_test:
    print(f"正在分析参数: {param}")

    # 准备数据
    X1, X2 = prepare_data_for_test(df, param, group1_id, group3_id)

    # 检查数据有效性
    if X1.shape[0] == 0 or X2.shape[0] == 0:
        print(f"警告：参数 '{param}' 的一个或两个组别没有数据，跳过此参数。")
        continue

    # 执行基于簇的置换检验
    # mne.stats.permutation_cluster_test适用于独立样本（组间比较）
    # tail=0表示双边检验
    t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
        [X1, X2],
        n_permutations=n_permutations,
        threshold=t_threshold,
        adjacency=adjacency,
        tail=0,
        n_jobs=-1  # 使用所有CPU核心并行计算，加快速度
    )

    # 找出显著的簇
    good_clusters_indices = np.where(cluster_p_values < p_threshold_monte_carlo)[0]

    print(f"结果: 共发现 {len(clusters)} 个簇。")
    if len(good_clusters_indices) > 0:
        print(f"发现 {len(good_clusters_indices)} 个显著簇 (p < {p_threshold_monte_carlo})。")
        for i, cluster_idx in enumerate(good_clusters_indices):
            cluster_p = cluster_p_values[cluster_idx]
            # cluster[cluster_idx] 是一个布尔掩码，标记了该簇中的电极
            ch_inds = np.where(clusters[cluster_idx])[0]
            cluster_chans = [ch_names[i] for i in ch_inds]
            print(f"  - 显著簇 {i + 1}: p = {cluster_p:.4f}, 包含电极: {cluster_chans}")
    else:
        print("未发现显著簇。")

    # --- 6. 可视化：绘制头皮拓扑图 ---
    # 准备用于在图上标记显著电极的掩码(mask)
    # 创建一个全为False的掩码
    sig_chans_mask = np.zeros(n_channels, dtype=bool)
    if len(good_clusters_indices) > 0:
        # 将所有显著簇中的电极位置在掩码中设为True
        for idx in good_clusters_indices:
            sig_chans_mask[clusters[idx]] = True

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 5))
    title = f't-values: {param} (Group {group1_id} vs {group3_id})'

    # 使用 mne.viz.plot_topomap 绘制t值拓扑图
    # t_obs 是每个电极的原始t值
    # sig_chans_mask 用于标记显著电极（会画出黑点）
    im, cn = mne.viz.plot_topomap(
        data=t_obs,
        pos=info,
        axes=ax,
        show=False,
        cmap='RdBu_r',  # 红-蓝配色，中间为0，与您的示例图类似
        mask=sig_chans_mask,
        mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4)
    )

    ax.set_title(title, fontweight='bold')

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('t-value')

    plt.show()
    print("-" * 50)