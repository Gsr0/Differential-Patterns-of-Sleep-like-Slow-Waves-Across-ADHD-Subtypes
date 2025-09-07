import pandas as pd
import numpy as np
import mne
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
import warnings

warnings.filterwarnings("ignore")

# --- 1. 参数设置与数据加载 ---
file_path = '所有慢波参数分电极总表.xlsx'
sheet_name = 'Sheet1'

# 电极名称列表
electrode_names = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
    'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5',
    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'
]

# 需要进行检验的参数列名
params_to_test = [
    'maxnegpkamp', 'mxdnslp'
]

# 组别定义
group1_id = 1  # ADHD 1型
group3_id = 3  # ADHD 3型

# 定义所有对比组合
comparisons = [
    (group1_id, group3_id, "ADHD1型 vs ADHD3型")
]

# 置换检验参数
p_threshold_cluster = 0.04  # 用于形成簇的p值阈值
p_threshold_monte_carlo = 0.05  # 用于判断簇是否显著的蒙特卡洛p值
n_permutations = 5000  # 置换次数

# 加载数据
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print("Excel文件加载成功。")
    print(f"数据包含 {df['SubjectID'].nunique()} 名被试。")

    # 打印各组的被试数量
    group_counts = df.groupby('Group')['SubjectID'].nunique()
    print("各组被试数量:")
    for group_id in [group1_id, group3_id]:
        if group_id in group_counts.index:
            print(f"  组 {group_id}: {group_counts[group_id]} 名被试")
        else:
            print(f"  组 {group_id}: 0 名被试 (警告：该组无数据)")

except FileNotFoundError:
    print(f"错误：未找到文件 '{file_path}'。请检查文件路径是否正确。")
    exit()

# --- 2. MNE准备工作：创建电极信息对象 ---
ch_numbers_in_data = sorted(df['Channel'].unique())
n_channels = len(ch_numbers_in_data)
ch_names = electrode_names[:n_channels]

print(f"\n数据中检测到 {n_channels} 个电极。")
print(f"将要使用的电极名称: {ch_names}")

# 创建MNE所需的info结构
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')

# 设置电极位置
montage = mne.channels.read_custom_montage('locations.sfp')
info.set_montage(montage, on_missing='warn')

# --- 3. 定义电极邻近关系 ---
adjacency, adj_ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')

if adj_ch_names != ch_names:
    print("警告：邻近矩阵的电极顺序与数据不匹配，正在尝试重新索引。")
    adj_indices = [adj_ch_names.index(ch) for ch in ch_names]
    adjacency = adjacency[np.ix_(adj_indices, adj_indices)]

print("\n电极邻近关系矩阵创建完成。")


# --- 4. LME模型函数定义 ---
def fit_lme_model(data, param_name, group1_val, group2_val):
    """
    为单个电极的数据拟合线性混合效应模型

    Parameters:
    -----------
    data : DataFrame
        包含SubjectID, Group, Age和参数值的数据框
    param_name : str
        要分析的参数名称
    group1_val : int
        第一组的组别值（作为参考组）
    group2_val : int
        第二组的组别值

    Returns:
    --------
    t_stat : float
        Group效应的t统计量
    p_value : float
        Group效应的p值
    """
    try:
        # 将Group转换为分类变量，以group1_val为参考组
        data = data.copy()
        data['Group'] = data['Group'].map({group1_val: 0, group2_val: 1})

        # 标准化Age以提高数值稳定性
        data['Age_std'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()

        # 拟合线性混合效应模型
        # 这里使用简单的随机截距模型，SubjectID作为随机效应
        formula = f"{param_name} ~ Group + Age_std"
        model = mixedlm(formula, data, groups=data['SubjectID'],
                        missing='drop')
        result = model.fit(method='lbfgs', maxiter=1000)

        # 提取Group效应的t统计量和p值
        group_idx = result.params.index.get_loc('Group')
        t_stat = result.tvalues[group_idx]
        p_value = result.pvalues[group_idx]

        return t_stat, p_value

    except Exception as e:
        print(f"拟合LME模型时出错: {e}")
        return np.nan, np.nan


def prepare_data_for_lme(df, param_name, group1, group2):
    """
    为LME模型准备数据

    Returns:
    --------
    data_list : list
        每个电极的数据框列表
    """
    # 筛选出当前要分析的参数和两个组别的数据
    df_param = df[df['Group'].isin([group1, group2])][
        ['SubjectID', 'Group', 'Age', 'Channel', param_name]
    ].dropna()

    data_list = []
    for ch in range(1, n_channels + 1):
        ch_data = df_param[df_param['Channel'] == ch].copy()
        if len(ch_data) > 0:
            data_list.append(ch_data)
        else:
            data_list.append(None)

    return data_list


# --- 5. 自定义LME置换检验函数 ---
def lme_permutation_cluster_test(data_list, param_name, adjacency, group1_val, group2_val,
                                 n_permutations=1000, p_threshold=0.025):
    """
    对LME模型进行基于簇的置换检验

    Parameters:
    -----------
    data_list : list
        每个电极的数据框列表
    param_name : str
        参数名称
    adjacency : sparse matrix
        电极邻近关系矩阵
    group1_val : int
        第一组的组别值
    group2_val : int
        第二组的组别值
    n_permutations : int
        置换次数
    p_threshold : float
        形成簇的p值阈值

    Returns:
    --------
    t_obs : array
        观察到的t统计量
    clusters : list
        簇的列表
    cluster_p_values : array
        簇的p值
    """
    n_channels = len(data_list)
    t_obs = np.zeros(n_channels)
    p_obs = np.zeros(n_channels)

    # 1. 计算观察到的统计量
    print("计算观察到的LME统计量...")
    for ch in range(n_channels):
        if data_list[ch] is not None:
            t_stat, p_val = fit_lme_model(data_list[ch], param_name, group1_val, group2_val)
            t_obs[ch] = t_stat
            p_obs[ch] = p_val
        else:
            t_obs[ch] = 0
            p_obs[ch] = 1

    # 2. 根据p值阈值确定显著电极
    threshold_mask = p_obs < p_threshold

    # 3. 基于邻近关系找到簇
    from scipy.sparse.csgraph import connected_components

    # 创建邻近矩阵的子集，只包含显著的电极
    if np.any(threshold_mask):
        sig_adjacency = adjacency.copy()
        sig_adjacency = sig_adjacency.multiply(
            np.outer(threshold_mask, threshold_mask)
        )

        # 找到连通分量（簇）
        n_components, labels = connected_components(
            sig_adjacency, directed=False
        )

        clusters = []
        cluster_stats = []

        for i in range(n_components):
            cluster_mask = (labels == i) & threshold_mask
            if np.sum(cluster_mask) > 0:
                clusters.append(cluster_mask)
                # 簇统计量为簇内t值的绝对值之和
                cluster_stat = np.sum(np.abs(t_obs[cluster_mask]))
                cluster_stats.append(cluster_stat)
    else:
        clusters = []
        cluster_stats = []

    if len(clusters) == 0:
        return t_obs, [], np.array([])

    cluster_stats = np.array(cluster_stats)

    # 4. 置换检验
    print(f"进行 {n_permutations} 次置换检验...")
    null_cluster_stats = []

    for perm in range(n_permutations):
        if (perm + 1) % 1000 == 0:
            print(f"  完成 {perm + 1}/{n_permutations} 次置换")

        # 为每个电极独立置换Group标签
        t_perm = np.zeros(n_channels)
        p_perm = np.zeros(n_channels)

        for ch in range(n_channels):
            if data_list[ch] is not None:
                data_perm = data_list[ch].copy()
                # 置换Group标签
                data_perm['Group'] = np.random.permutation(data_perm['Group'].values)
                t_stat_perm, p_val_perm = fit_lme_model(data_perm, param_name, group1_val, group2_val)
                t_perm[ch] = t_stat_perm
                p_perm[ch] = p_val_perm
            else:
                t_perm[ch] = 0
                p_perm[ch] = 1

        # 找到置换后的显著电极和簇
        perm_threshold_mask = p_perm < p_threshold

        if np.any(perm_threshold_mask):
            perm_sig_adjacency = adjacency.copy()
            perm_sig_adjacency = perm_sig_adjacency.multiply(
                np.outer(perm_threshold_mask, perm_threshold_mask)
            )

            perm_n_components, perm_labels = connected_components(
                perm_sig_adjacency, directed=False
            )

            perm_max_cluster_stat = 0
            for i in range(perm_n_components):
                perm_cluster_mask = (perm_labels == i) & perm_threshold_mask
                if np.sum(perm_cluster_mask) > 0:
                    perm_cluster_stat = np.sum(np.abs(t_perm[perm_cluster_mask]))
                    perm_max_cluster_stat = max(perm_max_cluster_stat, perm_cluster_stat)

            null_cluster_stats.append(perm_max_cluster_stat)
        else:
            null_cluster_stats.append(0)

    null_cluster_stats = np.array(null_cluster_stats)

    # 5. 计算簇的p值
    cluster_p_values = np.zeros(len(cluster_stats))
    for i, cluster_stat in enumerate(cluster_stats):
        cluster_p_values[i] = np.mean(null_cluster_stats >= cluster_stat)

    return t_obs, clusters, cluster_p_values


# --- 6. 主分析循环 ---
print(f"\n开始进行LME统计检验...")
print(f"用于形成簇的p值阈值: {p_threshold_cluster}")
print(f"用于判断簇显著性的蒙特卡洛p值阈值: {p_threshold_monte_carlo}")
print(f"将进行 {len(comparisons)} 种组间比较")
print("=" * 70)

for param in params_to_test:
    print(f"\n正在分析参数: {param}")
    print("=" * 50)

    for comp_idx, (group1, group2, comp_name) in enumerate(comparisons):
        print(f"\n比较 {comp_idx + 1}/{len(comparisons)}: {comp_name} (组{group1} vs 组{group2})")

        # 准备数据
        data_list = prepare_data_for_lme(df, param, group1, group2)

        # 检查数据有效性
        valid_channels = sum(1 for data in data_list if data is not None)
        if valid_channels == 0:
            print(f"警告：参数 '{param}' 在此组合中没有有效数据，跳过此比较。")
            continue

        # 检查每组的数据量
        total_subjects = set()
        group1_subjects = set()
        group2_subjects = set()

        for data in data_list:
            if data is not None:
                total_subjects.update(data['SubjectID'].unique())
                group1_subjects.update(data[data['Group'] == group1]['SubjectID'].unique())
                group2_subjects.update(data[data['Group'] == group2]['SubjectID'].unique())

        print(f"数据概况: 组{group1} {len(group1_subjects)}名被试, 组{group2} {len(group2_subjects)}名被试")

        if len(group1_subjects) < 5 or len(group2_subjects) < 5:
            print("警告：某组被试数量过少(<5)，结果可能不可靠。")

        # 执行LME置换检验
        t_obs, clusters, cluster_p_values = lme_permutation_cluster_test(
            data_list, param, adjacency, group1, group2,
            n_permutations=n_permutations,
            p_threshold=p_threshold_cluster
        )

        # 找出显著的簇
        good_clusters_indices = np.where(cluster_p_values < p_threshold_monte_carlo)[0]

        print(f"结果: 共发现 {len(clusters)} 个簇。")
        if len(good_clusters_indices) > 0:
            print(f"发现 {len(good_clusters_indices)} 个显著簇 (p < {p_threshold_monte_carlo})。")
            for i, cluster_idx in enumerate(good_clusters_indices):
                cluster_p = cluster_p_values[cluster_idx]
                ch_inds = np.where(clusters[cluster_idx])[0]
                cluster_chans = [ch_names[i] for i in ch_inds]
                print(f"  - 显著簇 {i + 1}: p = {cluster_p:.4f}, 包含电极: {cluster_chans}")
        else:
            print("未发现显著簇。")

        # --- 7. 可视化：绘制头皮拓扑图 ---
        sig_chans_mask = np.zeros(n_channels, dtype=bool)
        if len(good_clusters_indices) > 0:
            for idx in good_clusters_indices:
                sig_chans_mask[clusters[idx]] = True

        fig, ax = plt.subplots(figsize=(6, 5))
        title = f'LME t-values: {param}\n{comp_name} (Age adjusted)'

        im, cn = mne.viz.plot_topomap(
            data=t_obs,
            pos=info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            mask=sig_chans_mask,
            mask_params=dict(marker='o', markerfacecolor='k',
                             markeredgecolor='k', linewidth=0, markersize=4)
        )

        ax.set_title(title, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('t-value (LME)')

        plt.tight_layout()
        plt.show()

        print("-" * 30)

    print("=" * 50)

print("\n所有分析完成！")
print("总结:")
print(f"- 分析了 {len(params_to_test)} 个参数")
print(f"- 进行了 {len(comparisons)} 种组间比较")
print(f"- 每个比较使用了 {n_permutations} 次置换检验")