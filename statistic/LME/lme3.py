import pandas as pd
import numpy as np
import mne
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore')

# --- 1. 参数设置与数据加载 ---
# 请将 'your_data_file.xlsx' 替换为您的Excel文件实际路径
file_path = '所有慢波参数分电极总表.xlsx'
# 请将 'Sheet1' 替换为您的Excel中的工作表名称
sheet_name = 'Sheet1'

# 您提供的电极名称列表 (共31个，但实际使用30个)
electrode_names = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
    'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5',
    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8','POz', 'O1','Oz', 'O2'
]

# 需要进行检验的参数列名
params_to_test = [
    'maxnegpkamp', 'maxpospkamp', 'mxdnslp', 'mxupslp', 'sw_density', 'mean_duration'
]

# 组别定义
group1_id = 1  # ADHD 1型
group3_id = 3  # ADHD 3型

# 置换检验参数
p_threshold_cluster = 0.025  # 用于形成簇的p值阈值
p_threshold_monte_carlo = 0.05  # 用于判断簇是否显著的蒙特卡洛p值
n_permutations = 5000  # 置换次数

# 加载数据
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print("Excel文件加载成功。")
    print(f"数据包含 {df['SubjectID'].nunique()} 名被试。")
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


# --- 4. LME分析函数 ---
def run_lme_analysis(df, param_name, group1, group2):
    """
    使用线性混合效应模型进行分析，以年龄为协变量
    返回每个电极的t值和p值
    """
    # 筛选出需要的组别和参数
    df_analysis = df[df['Group'].isin([group1, group2])].copy()

    # 确保年龄列存在
    if 'Age' not in df_analysis.columns:
        print(f"警告：数据中没有找到'Age'列，将不使用年龄作为协变量")
        use_age = False
    else:
        use_age = True

    t_values = []
    p_values = []

    # 对每个电极分别进行LME分析
    for channel in sorted(df_analysis['Channel'].unique()):
        try:
            # 提取当前电极的数据
            channel_data = df_analysis[df_analysis['Channel'] == channel].copy()

            # 检查数据完整性
            if channel_data[param_name].isna().any():
                print(f"警告：电极 {channel} 的参数 {param_name} 存在缺失值，跳过")
                t_values.append(0)
                p_values.append(1)
                continue

            # 准备LME模型的公式
            if use_age:
                formula = f"{param_name} ~ Group + Age"
            else:
                formula = f"{param_name} ~ Group"

            # 拟合LME模型（以SubjectID作为随机效应）
            # 注意：由于每个被试在每个电极只有一个观测值，这里实际上是普通的线性回归
            # 但我们保持LME的框架以便将来扩展
            try:
                model = mixedlm(formula, channel_data, groups=channel_data["SubjectID"])
                result = model.fit(method='lbfgs')

                # 提取组别效应的t值和p值
                group_coef_name = 'Group'
                if group_coef_name in result.params.index:
                    t_val = result.tvalues[group_coef_name]
                    p_val = result.pvalues[group_coef_name]
                else:
                    # 如果找不到Group系数，可能是编码问题，尝试其他可能的名称
                    group_params = [param for param in result.params.index if 'Group' in str(param)]
                    if group_params:
                        t_val = result.tvalues[group_params[0]]
                        p_val = result.pvalues[group_params[0]]
                    else:
                        t_val = 0
                        p_val = 1

                t_values.append(t_val)
                p_values.append(p_val)

            except Exception as e:
                print(f"LME模型拟合失败，电极 {channel}: {str(e)}")
                # 如果LME失败，回退到普通t检验
                group1_data = channel_data[channel_data['Group'] == group1][param_name]
                group2_data = channel_data[channel_data['Group'] == group2][param_name]

                if len(group1_data) > 0 and len(group2_data) > 0:
                    t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
                    t_values.append(t_stat)
                    p_values.append(p_val)
                else:
                    t_values.append(0)
                    p_values.append(1)

        except Exception as e:
            print(f"分析电极 {channel} 时出错: {str(e)}")
            t_values.append(0)
            p_values.append(1)

    return np.array(t_values), np.array(p_values)


# --- 5. 基于LME结果的簇检验函数 ---
def lme_cluster_permutation_test(df, param_name, group1, group2, adjacency,
                                 n_permutations=1000, p_threshold=0.025):
    """
    基于LME分析结果进行簇置换检验
    """
    # 获取原始的LME分析结果
    original_t, original_p = run_lme_analysis(df, param_name, group1, group2)

    # 创建簇（基于p值阈值）
    significant_mask = original_p < p_threshold

    # 如果没有显著的电极，返回空结果
    if not np.any(significant_mask):
        return original_t, [], [], []

    # 使用MNE的簇检测算法
    # 先将t值转换为与p值阈值对应的t阈值
    df_residual = len(df['SubjectID'].unique()) - 2  # 自由度估计
    t_threshold = stats.t.ppf(1.0 - p_threshold / 2, df=df_residual)

    # 创建伪数据进行置换检验
    df_perm = df[df['Group'].isin([group1, group2])].copy()

    # 存储置换结果
    cluster_stats = []

    # 检测原始数据中的簇
    clusters = []
    cluster_t_sums = []

    # 简化的簇检测：基于邻接矩阵找连通的显著区域
    visited = np.zeros(len(original_t), dtype=bool)

    for i in range(len(original_t)):
        if significant_mask[i] and not visited[i]:
            # 开始一个新簇
            cluster = []
            stack = [i]

            while stack:
                current = stack.pop()
                if visited[current]:
                    continue

                visited[current] = True
                cluster.append(current)

                # 检查相邻电极
                for j in range(len(original_t)):
                    if (not visited[j] and significant_mask[j] and
                            adjacency[current, j]):
                        stack.append(j)

            if cluster:
                clusters.append(cluster)
                cluster_t_sums.append(np.sum(np.abs(original_t[cluster])))

    # 置换检验
    null_distribution = []

    for perm in range(n_permutations):
        # 随机置换组标签
        df_shuffled = df_perm.copy()
        subjects = df_shuffled['SubjectID'].unique()
        np.random.shuffle(subjects)

        # 创建置换后的组标签映射
        n_group1 = len(df_perm[df_perm['Group'] == group1]['SubjectID'].unique())
        group1_subjects = subjects[:n_group1]
        group2_subjects = subjects[n_group1:]

        # 应用置换
        df_shuffled.loc[df_shuffled['SubjectID'].isin(group1_subjects), 'Group'] = group1
        df_shuffled.loc[df_shuffled['SubjectID'].isin(group2_subjects), 'Group'] = group2

        # 计算置换后的统计量
        perm_t, perm_p = run_lme_analysis(df_shuffled, param_name, group1, group2)
        perm_significant = perm_p < p_threshold

        # 检测置换数据中的最大簇统计量
        max_cluster_stat = 0
        visited_perm = np.zeros(len(perm_t), dtype=bool)

        for i in range(len(perm_t)):
            if perm_significant[i] and not visited_perm[i]:
                cluster_perm = []
                stack = [i]

                while stack:
                    current = stack.pop()
                    if visited_perm[current]:
                        continue

                    visited_perm[current] = True
                    cluster_perm.append(current)

                    for j in range(len(perm_t)):
                        if (not visited_perm[j] and perm_significant[j] and
                                adjacency[current, j]):
                            stack.append(j)

                if cluster_perm:
                    cluster_stat = np.sum(np.abs(perm_t[cluster_perm]))
                    max_cluster_stat = max(max_cluster_stat, cluster_stat)

        null_distribution.append(max_cluster_stat)

    # 计算簇的p值
    cluster_p_values = []
    for cluster_stat in cluster_t_sums:
        p_val = np.mean([null_stat >= cluster_stat for null_stat in null_distribution])
        cluster_p_values.append(p_val)

    return original_t, clusters, cluster_p_values, cluster_t_sums


# --- 6. 数据整理函数（保留原有的t检验方法作为备选） ---
def prepare_data_for_test(df, param_name, group1, group2):
    """
    将数据从长格式整理为宽格式，用于传统的置换检验
    """
    df_param = df[df['Group'].isin([group1, group2])][['SubjectID', 'Group', 'Channel', param_name]]
    df_pivot = df_param.pivot_table(index=['SubjectID', 'Group'], columns='Channel', values=param_name)

    data_g1 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group1].values
    data_g3 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group2].values

    return data_g1, data_g3


# --- 7. 主分析循环 ---
print(f"\n开始进行LME统计检验...")
print(f"用于形成簇的p值阈值: {p_threshold_cluster}")
print(f"用于判断簇显著性的蒙特卡洛p值阈值: {p_threshold_monte_carlo}")
print("-" * 50)

# 选择分析方法
USE_LME = True  # 设置为True使用LME方法，False使用原有的t检验方法

for param in params_to_test:
    print(f"正在分析参数: {param}")

    if USE_LME:
        # 使用LME方法
        print("使用线性混合效应模型(LME)进行分析...")

        # 进行LME簇置换检验
        t_obs, clusters, cluster_p_values, cluster_stats = lme_cluster_permutation_test(
            df, param, group1_id, group3_id, adjacency, n_permutations, p_threshold_cluster
        )

        # 找出显著的簇
        good_clusters_indices = np.where(np.array(cluster_p_values) < p_threshold_monte_carlo)[0]

    else:
        # 使用原有的t检验方法
        print("使用传统的t检验方法...")

        # 准备数据
        X1, X2 = prepare_data_for_test(df, param, group1_id, group3_id)

        if X1.shape[0] == 0 or X2.shape[0] == 0:
            print(f"警告：参数 '{param}' 的一个或两个组别没有数据，跳过此参数。")
            continue

        # 计算t阈值
        t_threshold = stats.t.ppf(1.0 - p_threshold_cluster / 2, df=df['SubjectID'].nunique() - 2)

        # 执行传统的基于簇的置换检验
        t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
            [X1, X2],
            n_permutations=n_permutations,
            threshold=t_threshold,
            adjacency=adjacency,
            tail=0,
            n_jobs=-1
        )

        good_clusters_indices = np.where(cluster_p_values < p_threshold_monte_carlo)[0]

    # 输出结果
    print(f"结果: 共发现 {len(clusters)} 个簇。")
    if len(good_clusters_indices) > 0:
        print(f"发现 {len(good_clusters_indices)} 个显著簇 (p < {p_threshold_monte_carlo})。")
        for i, cluster_idx in enumerate(good_clusters_indices):
            if USE_LME:
                cluster_p = cluster_p_values[cluster_idx]
                ch_inds = clusters[cluster_idx]
            else:
                cluster_p = cluster_p_values[cluster_idx]
                ch_inds = np.where(clusters[cluster_idx])[0]

            cluster_chans = [ch_names[i] for i in ch_inds]
            print(f"  - 显著簇 {i + 1}: p = {cluster_p:.4f}, 包含电极: {cluster_chans}")
    else:
        print("未发现显著簇。")

    # --- 8. 可视化：绘制头皮拓扑图 ---
    sig_chans_mask = np.zeros(n_channels, dtype=bool)
    if len(good_clusters_indices) > 0:
        for idx in good_clusters_indices:
            if USE_LME:
                sig_chans_mask[clusters[idx]] = True
            else:
                sig_chans_mask[clusters[idx]] = True

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    if USE_LME:
        title = f'LME t-values: {param} (Group {group1_id} vs {group3_id})\nwith Age as covariate'
    else:
        title = f'T-test t-values: {param} (Group {group1_id} vs {group3_id})'

    im, cn = mne.viz.plot_topomap(
        data=t_obs,
        pos=info,
        axes=ax,
        show=False,
        cmap='RdBu_r',
        mask=sig_chans_mask,
        mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                         linewidth=0, markersize=6)
    )

    ax.set_title(title, fontweight='bold')

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('t-value')

    plt.tight_layout()
    plt.show()

    # 如果使用LME，还可以输出更详细的统计信息
    if USE_LME:
        # 进行多重比较校正
        _, p_corrected, _, _ = multipletests(
            run_lme_analysis(df, param, group1_id, group3_id)[1],
            method='fdr_bh'
        )

        print(f"多重比较校正后的显著电极（FDR < 0.05）:")
        sig_electrodes_corrected = np.where(p_corrected < 0.05)[0]
        if len(sig_electrodes_corrected) > 0:
            for idx in sig_electrodes_corrected:
                print(f"  - {ch_names[idx]}: t = {t_obs[idx]:.3f}, p_corrected = {p_corrected[idx]:.4f}")
        else:
            print("  无显著电极")

    print("-" * 50)

print("\n分析完成！")


# --- 9. 补充：保存结果到Excel ---
def save_results_to_excel(df, params_to_test, group1_id, group3_id, filename='LME_analysis_results.xlsx'):
    """
    将LME分析结果保存到Excel文件
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for param in params_to_test:
            t_values, p_values = run_lme_analysis(df, param, group1_id, group3_id)

            # 创建结果DataFrame
            results_df = pd.DataFrame({
                'Channel': range(1, len(t_values) + 1),
                'Electrode': ch_names[:len(t_values)],
                't_value': t_values,
                'p_value': p_values
            })

            # 添加多重比较校正
            _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            results_df['p_corrected_FDR'] = p_corrected
            results_df['significant_uncorrected'] = p_values < 0.05
            results_df['significant_FDR'] = p_corrected < 0.05

            # 保存到不同的sheet
            results_df.to_excel(writer, sheet_name=param, index=False)

    print(f"结果已保存到 {filename}")

# 如果需要保存结果，取消下面这行的注释
# save_results_to_excel(df, params_to_test, group1_id, group3_id)