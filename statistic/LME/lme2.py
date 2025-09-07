import pandas as pd
import numpy as np
import mne
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # 导入 statsmodels 用于线性混合模型

# --- 1. 参数设置与数据加载 ---
# 请将 'your_data_file.xlsx' 替换为您的Excel文件实际路径
file_path = '所有慢波参数分电极总表.xlsx'
# 请将 'Sheet1' 替换为您的Excel中的工作表名称
sheet_name = 'Sheet1'

# 您提供的电极名称列表 (共31个)
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
group0_id = 0  # 正常对照组
group1_id = 1  # ADHD 1型
group3_id = 3  # ADHD 3型

# 定义要进行的所有组间比较
# 每个元组：(组A的ID, 组B的ID, 用于绘图的比较标签)
comparison_groups = [
    (group1_id, group3_id, f'Group {group1_id} vs {group3_id}'),
    (group0_id, group1_id, f'Group {group0_id} vs {group1_id}'),
    (group0_id, group3_id, f'Group {group0_id} vs {group3_id}')
]

# 置换检验参数
p_threshold_cluster = 0.025  # 用于形成簇的p值阈值 (双边，即单边为0.0125)
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

# --- IMPORTANT: 为 LME 演示添加 'Age' 列 ---
# 在实际应用中，您应该确保您的Excel数据中包含真实的 'Age' 列。
# 如果您的数据中没有 'Age' 列，以下代码将生成一个随机的 'Age' 列作为演示。
if 'Age' not in df.columns:
    print("\n警告：未找到 'Age' 列。为演示LME，已添加随机生成的分组年龄数据。")
    print("请确保您的实际数据包含 'Age' 列以获得有意义的LME结果。")
    np.random.seed(42) # 设置随机种子以保证结果可复现

    # 为每个被试分配一个基于其组别的随机年龄
    # 模拟不同组可能略有不同的年龄分布
    unique_subjects = df['SubjectID'].unique()
    subject_age_map = {}
    for sub_id in unique_subjects:
        # 获取该被试所属的组别
        group_for_sub = df[df['SubjectID'] == sub_id]['Group'].iloc[0]
        if group_for_sub == group0_id: # 正常对照组
            subject_age_map[sub_id] = np.random.randint(18, 25) # 例如，18-24岁
        elif group_for_sub == group1_id: # ADHD 1型
            subject_age_map[sub_id] = np.random.randint(19, 28) # 例如，19-27岁
        else: # ADHD 3型
            subject_age_map[sub_id] = np.random.randint(20, 29) # 例如，20-28岁
    df['Age'] = df['SubjectID'].map(subject_age_map)
    print("随机 'Age' 列添加完成。")

# 确保 'SubjectID' 列被视为分类变量
df['SubjectID'] = df['SubjectID'].astype('category')


# --- 2. MNE准备工作：创建电极信息对象 ---
# 从数据中获取实际使用的电极数量和编号
ch_numbers_in_data = sorted(df['Channel'].unique())
n_channels = len(ch_numbers_in_data)
# 根据数据中的电极数量，从您提供的列表中获取对应的电极名称
ch_names = electrode_names[:n_channels]

print(f"\n数据中检测到 {n_channels} 个电极。")
print(f"将要使用的电极名称: {ch_names}")

# 创建MNE所需的info结构，它包含了电极的基本信息
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg') # sfreq可以任意设置，因为这里不分析时序数据

# 设置电极位置。MNE会自动从其内置的标准10-20系统中寻找电极位置
montage = mne.channels.read_custom_montage('locations.sfp')
info.set_montage(montage, on_missing='warn') # on_missing='warn' 会对找不到位置的电极发出警告

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


# --- 4. 数据整理：将长格式数据转换为适合置换检验的宽格式 ---
def prepare_data_for_permutation_test(df_input, param_name, group1, group2):
    """
    将数据从长格式整理为宽格式，并按组别分开，用于置换检验。
    返回两个数组，分别是两个组的数据，形状为 (n_subjects, n_channels)。
    """
    # 筛选出当前要分析的参数和两个组别的数据
    df_param = df_input[df_input['Group'].isin([group1, group2])][['SubjectID', 'Group', 'Channel', param_name]]

    # 数据透视/转换，将每个电极的参数值转换为列
    df_pivot = df_param.pivot_table(index=['SubjectID', 'Group'], columns='Channel', values=param_name)

    # 确保 pivot table 的列顺序与 ch_names 对应的通道编号一致
    # 假设 df['Channel'] 包含 1-based 编号，且与 electrode_names 的顺序匹配
    # 如果数据中某些通道缺失，这里会自动填充 NaN
    df_pivot = df_pivot.reindex(columns=range(1, len(ch_names) + 1), fill_value=np.nan)

    # 分离两个组的数据并转换为 numpy 数组
    data_g1 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group1].values
    data_g2 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group2].values

    return data_g1, data_g2


# --- 新增方法：独立样本 t 检验 ---
def run_independent_ttest(data_df, param_name, group1_id, group2_id, ch_names):
    """
    对每个电极执行独立样本 t 检验。
    返回一个包含 t 值和 p 值的 numpy 数组，顺序与 ch_names 一致。
    """
    # 初始化存储结果的数组，用 NaN 填充
    t_values = np.full(len(ch_names), np.nan)
    p_values = np.full(len(ch_names), np.nan)

    # 筛选出当前要分析的参数和两个组别的数据
    df_filtered = data_df[data_df['Group'].isin([group1_id, group2_id])][['SubjectID', 'Group', 'Channel', param_name]]

    # 遍历每个电极
    for i, ch_name in enumerate(ch_names):
        # 假设 'Channel' 列中的编号是从 1 开始，并且与 ch_names 的顺序对应 (即 ch_names[0] 对应 Channel 1)
        ch_num = i + 1

        # 提取当前电极、当前参数在两个组的数据
        data_g1_ch = df_filtered[(df_filtered['Channel'] == ch_num) & (df_filtered['Group'] == group1_id)][param_name].dropna()
        data_g2_ch = df_filtered[(df_filtered['Channel'] == ch_num) & (df_filtered['Group'] == group2_id)][param_name].dropna()

        # 确保每个组至少有两个有效数据点才能进行 t 检验
        if len(data_g1_ch) > 1 and len(data_g2_ch) > 1:
            # 执行独立样本 t 检验
            # equal_var=True 假设方差相等 (Levene检验可用于检查)
            t_stat, p_val = stats.ttest_ind(data_g1_ch, data_g2_ch, equal_var=True)
            t_values[i] = t_stat
            p_values[i] = p_val
        # else:
        #     # 如果数据不足，则对应的 t 值和 p 值保持为 NaN
        #     print(f"警告：电极 {ch_name} (编号{ch_num}) 在一个或两个组别中数据不足，无法进行独立样本t检验。")

    return t_values, p_values


# --- 新增方法：线性混合模型 (LME) ---
def run_lme_per_channel(data_df, param_name, group1_id, group2_id, ch_names):
    """
    对每个电极，使用年龄作为协变量，执行线性混合模型。
    返回一个包含每个电极的组间比较 (group effect) 的 Z 值和 P 值，顺序与 ch_names 一致。
    Z 值在 LME 中类似于 t 值，用于衡量效应的显著性。
    """
    # 初始化存储结果的数组，用 NaN 填充
    z_values = np.full(len(ch_names), np.nan)
    p_values = np.full(len(ch_names), np.nan)

    # 筛选出当前要分析的参数、两个组别、年龄和 SubjectID 的数据
    df_lme = data_df[data_df['Group'].isin([group1_id, group2_id])][['SubjectID', 'Group', 'Channel', 'Age', param_name]].copy()

    # 确保 'SubjectID' 为分类变量
    df_lme['SubjectID'] = df_lme['SubjectID'].astype('category')

    # 关键修改：确保 'Group' 列的类别只包含当前比较的两个组，并设置正确的类别顺序
    # 这样 group1_id 会成为 statsmodels 中的参考水平
    df_lme['Group'] = pd.Categorical(df_lme['Group'], categories=[group1_id, group2_id])

    # 遍历每个电极
    for i, ch_name in enumerate(ch_names):
        ch_num = i + 1 # 假设 'Channel' 列中的编号是从 1 开始

        # 提取当前电极的数据
        df_channel = df_lme[df_lme['Channel'] == ch_num].copy()

        # 检查当前电极在两个组中是否有足够的有效数据
        # 至少需要每个组有2个被试才能拟合LME
        count_g1 = df_channel[df_channel['Group'] == group1_id].shape[0]
        count_g2 = df_channel[df_channel['Group'] == group2_id].shape[0]

        if count_g1 < 2 or count_g2 < 2:
            # print(f"警告：电极 {ch_name} (编号{ch_num}) 在一个或两个组别中数据不足，无法进行LME。")
            continue

        try:
            # 定义 LME 模型公式：
            # param_name ~ C(Group) + Age
            #   - param_name: 因变量（您要分析的慢波参数）
            #   - C(Group): 将 Group 作为分类变量处理（statsmodels会自动创建虚拟变量）
            #   - Age: 连续型协变量
            # groups=df_channel['SubjectID']: 指定随机效应的组（每个 SubjectID 有一个随机效应）
            # re_formula='1': 表示为每个 SubjectID 添加一个随机截距
            model_formula = f'{param_name} ~ C(Group) + Age'
            model = smf.mixedlm(model_formula, data=df_channel,
                                groups=df_channel['SubjectID'],
                                re_formula='1')

            # 拟合模型，maxiter 增加迭代次数以帮助收敛，disp=False 抑制详细的拟合过程输出
            fit_result = model.fit(maxiter=1000, disp=False)

            # 提取组间比较的统计量（Z值和P值）
            # 这里的项名称 'C(Group)[T.{group2_id}]' 表示 group2_id 相对于参考组 group1_id 的对比
            group_term_name = f'C(Group)[T.{group2_id}]'

            if group_term_name in fit_result.pvalues:
                p_values[i] = fit_result.pvalues[group_term_name]
                z_values[i] = fit_result.tvalues[group_term_name] # LME 通常用 Z 值表示，但 fit_result.tvalues 提供了类似 Z 值的统计量
            # else:
            #     print(f"警告：电极 {ch_name} (编号{ch_num}) LME结果中未找到组比较项 '{group_term_name}'。可能存在收敛问题或数据异常。")

        except Exception as e:
            # 捕获 LME 拟合过程中可能发生的错误 (如不收敛)
            # print(f"错误：电极 {ch_name} (编号{ch_num}) 的LME模型未能收敛或出现错误: {e}")
            pass # 继续处理下一个电极

    return z_values, p_values


# --- 5. 循环执行所有检验并可视化 ---

print(f"\n开始进行统计检验...")
print(f"用于形成簇的p值阈值: {p_threshold_cluster} (双边)")
print(f"用于判断簇显著性的蒙特卡洛p值阈值: {p_threshold_monte_carlo}")
print("-" * 50)

# 遍历每一个需要检验的参数
for param in params_to_test:
    # 遍历所有定义的组间比较
    for g1_id, g2_id, comparison_label in comparison_groups:
        print(f"\n--- 正在分析参数: {param} (比较: {comparison_label}) ---")

        # --- A. 基于簇的置换检验 (MNE) ---
        print("\n=== 基于簇的置换检验 (MNE) ===")
        # 准备数据，将其从长格式转换为宽格式
        X1, X2 = prepare_data_for_permutation_test(df, param, g1_id, g2_id)

        if X1.shape[0] == 0 or X2.shape[0] == 0:
            print(f"警告：参数 '{param}'，比较 '{comparison_label}' 的一个或两个组别没有数据，跳过此置换检验。")
            continue

        # 根据当前比较中的被试数量计算自由度，用于确定 t 值阈值
        n_subjects_in_comparison = X1.shape[0] + X2.shape[0]
        df_t_test = n_subjects_in_comparison - 2
        # 防止自由度为负或过小导致错误
        if df_t_test <= 0:
            print(f"警告：比较 '{comparison_label}' 的被试数量不足以进行有效的 t 检验 (自由度 <= 0)。跳过置换检验。")
            continue

        t_threshold = stats.t.ppf(1.0 - p_threshold_cluster / 2, df=df_t_test)
        print(f"当前比较 ({comparison_label}) 用于形成簇的 t 值阈值: {t_threshold:.3f}")

        # 执行基于簇的置换检验
        t_obs_mne, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
            [X1, X2],
            n_permutations=n_permutations,
            threshold=t_threshold,
            adjacency=adjacency,
            tail=0, # 双边检验
            n_jobs=-1 # 使用所有CPU核心并行计算
        )

        # 找出显著的簇
        good_clusters_indices = np.where(cluster_p_values < p_threshold_monte_carlo)[0]

        print(f"结果: 共发现 {len(clusters)} 个簇。")
        if len(good_clusters_indices) > 0:
            print(f"发现 {len(good_clusters_indices)} 个显著簇 (p < {p_threshold_monte_carlo})。")
            for i, cluster_idx in enumerate(good_clusters_indices):
                cluster_p = cluster_p_values[cluster_idx]
                # clusters[cluster_idx] 是一个布尔掩码，标记了该簇中的电极
                ch_inds = np.where(clusters[cluster_idx])[0]
                cluster_chans = [ch_names[i] for i in ch_inds]
                print(f"  - 显著簇 {i + 1}: p = {cluster_p:.4f}, 包含电极: {cluster_chans}")
        else:
            print("未发现显著簇。")

        # 可视化 MNE 结果
        fig, ax = plt.subplots(figsize=(6, 5))
        title_mne = f'Permutation Cluster Test (t-values): {param}\n({comparison_label})'
        # 准备用于在图上标记显著电极的掩码 (mask)
        sig_chans_mask_mne = np.zeros(n_channels, dtype=bool)
        if len(good_clusters_indices) > 0:
            # 将所有显著簇中的电极位置在掩码中设为True
            for idx in good_clusters_indices:
                sig_chans_mask_mne[clusters[idx]] = True

        im, cn = mne.viz.plot_topomap(
            data=t_obs_mne, # 绘制每个电极的原始 t 值
            pos=info,
            axes=ax,
            show=False,
            cmap='RdBu_r', # 红-蓝配色，中间为0
            mask=sig_chans_mask_mne, # 标记显著电极
            mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4)
        )
        ax.set_title(title_mne, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('t-value')
        plt.show()

        # --- B. 独立样本 t 检验 ---
        print("\n=== 独立样本 t 检验 ===")
        # 调用新定义的函数执行独立 t 检验
        t_obs_ttest, p_values_ttest = run_independent_ttest(df, param, g1_id, g2_id, ch_names)

        # 找出显著电极 (例如，p < 0.05)
        sig_chans_mask_ttest = p_values_ttest < 0.05

        print(f"发现 {np.sum(sig_chans_mask_ttest)} 个在独立t检验中显著的电极 (p < 0.05)。")
        if np.sum(sig_chans_mask_ttest) > 0:
            sig_ttest_chans = [ch_names[i] for i, is_sig in enumerate(sig_chans_mask_ttest) if is_sig]
            print(f"  显著电极: {sig_ttest_chans}")
        else:
            print("未发现显著电极。")

        # 可视化独立t检验结果
        fig, ax = plt.subplots(figsize=(6, 5))
        title_ttest = f'Independent t-test (t-values): {param}\n({comparison_label})'
        im, cn = mne.viz.plot_topomap(
            data=t_obs_ttest, # 绘制 t 值
            pos=info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            mask=sig_chans_mask_ttest, # 标记显著电极
            mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4)
        )
        ax.set_title(title_ttest, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('t-value')
        plt.show()

        # --- C. 线性混合模型 (LME) ---
        print("\n=== 线性混合模型 (LME) - 包含年龄协变量 ===")
        # 调用新定义的函数执行 LME
        z_obs_lme, p_values_lme = run_lme_per_channel(df, param, g1_id, g2_id, ch_names)

        # 找出显著电极 (例如，p < 0.05)
        sig_chans_mask_lme = p_values_lme < 0.05

        print(f"发现 {np.sum(sig_chans_mask_lme)} 个在LME中显著的电极 (p < 0.05)。")
        if np.sum(sig_chans_mask_lme) > 0:
            sig_lme_chans = [ch_names[i] for i, is_sig in enumerate(sig_chans_mask_lme) if is_sig]
            print(f"  显著电极: {sig_lme_chans}")
        else:
            print("未发现显著电极。")

        # 可视化 LME 结果
        fig, ax = plt.subplots(figsize=(6, 5))
        title_lme = f'LME (z-values, with Age): {param}\n({comparison_label})'
        im, cn = mne.viz.plot_topomap(
            data=z_obs_lme, # 绘制 LME 的 Z 值
            pos=info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            mask=sig_chans_mask_lme, # 标记显著电极
            mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4)
        )
        ax.set_title(title_lme, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('z-value')
        plt.show()
        print("-" * 50)

print("\n所有分析完成。")