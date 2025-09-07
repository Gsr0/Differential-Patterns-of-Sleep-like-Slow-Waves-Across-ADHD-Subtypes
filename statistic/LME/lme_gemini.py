import pandas as pd
import numpy as np
import mne
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
from mne.stats import permutation_cluster_1samp_test
from tqdm import tqdm
import warnings

# 忽略一些statsmodels和mne中可能出现的无害警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. 参数设置 ---
file_path = '所有慢波参数分电极总表.xlsx'
sheet_name = 'Sheet1'
electrode_layout_file = 'locations.sfp'  # 您的电极位置文件

electrode_names = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
    'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5',
    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'
]

params_to_test = [
    'maxnegpkamp', 'maxpospkamp', 'mxdnslp', 'mxupslp', 'sw_density', 'mean_duration'
]

# 定义所有需要进行的组别比较
# 格式: ('比较名称', 组A_ID, 组B_ID)
comparisons = [
    ('Group_1_vs_3', 1, 3),
    ('Group_0_vs_1', 0, 1),
    ('Group_0_vs_3', 0, 3)
]

# 置换检验参数
p_threshold_cluster = 0.05
n_permutations = 5000  # 建议至少1000，5000更稳定
montecarlo_alpha = 0.05

# --- 2. 数据与MNE对象加载 ---
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print("Excel文件加载成功。")
    print(f"数据包含 {df['SubjectID'].nunique()} 名被试。")
except FileNotFoundError:
    print(f"错误：未找到文件 '{file_path}'。")
    exit()

df['SubjectID'] = df['SubjectID'].astype('category')
df['Group'] = df['Group'].astype('category')

n_channels = len(df['Channel'].unique())
ch_names = electrode_names[:n_channels]
print(f"\n数据中检测到 {n_channels} 个电极: {ch_names}")

info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
montage = mne.channels.read_custom_montage(electrode_layout_file)
info.set_montage(montage)
adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type='eeg')
print("\n电极邻近关系矩阵创建完成。")


# --- 3. 稳健的LME拟合函数 ---
def run_lme_on_channel(data, param_name, group_a_id, group_b_id):
    """为单个电极的数据稳健地拟合LME模型"""
    df_ch = data.copy()
    try:
        # 将当前比较的两个组别映射为 0 和 1
        df_ch['Group_coded'] = df_ch['Group'].map({group_a_id: 0, group_b_id: 1})
        # 标准化年龄
        df_ch['Age_std'] = (df_ch['Age'] - df_ch['Age'].mean()) / df_ch['Age'].std()

        formula = f"{param_name} ~ Group_coded + Age_std"
        model = mixedlm(formula, df_ch, groups=df_ch['SubjectID'])
        result = model.fit(method='lbfgs', maxiter=2000)

        t_stat = result.tvalues['Group_coded']
        return t_stat
    except Exception:
        return 0.0  # 如果模型失败，返回0


# --- 4. 主分析循环 ---
for param in params_to_test:
    for comp_name, group_a, group_b in comparisons:

        print("\n" + "=" * 80)
        print(f"  分析参数: {param}  |  比较: {comp_name} (Group {group_a} vs {group_b})")
        print("=" * 80)

        # 4.1 准备当前比较所需的数据
        df_comp = df[df['Group'].isin([group_a, group_b])].copy()
        df_comp['Group'] = df_comp['Group'].cat.remove_unused_categories()

        if df_comp['SubjectID'].nunique() < 2:
            print("数据不足，跳过此比较。")
            continue

        # 4.2 计算真实的t值
        print("步骤 1/3: 计算真实的LME模型t值...")
        t_obs = np.zeros(n_channels)
        for i, ch_num in enumerate(tqdm(range(1, n_channels + 1), desc="计算真实t值")):
            ch_data = df_comp[df_comp['Channel'] == ch_num]
            if not ch_data.empty:
                t_obs[i] = run_lme_on_channel(ch_data, param, group_a, group_b)

        # 4.3 执行置换检验
        print(f"\n步骤 2/3: 执行 {n_permutations} 次置换检验...")

        subject_info = df_comp[['SubjectID', 'Group']].drop_duplicates()

        # 自由度近似值，用于计算t阈值
        dof = subject_info.shape[0] - 2
        t_threshold = stats.t.ppf(1 - p_threshold_cluster / 2, df=dof)
        print(f"用于形成簇的t值阈值 (双边p<{p_threshold_cluster}): {t_threshold:.3f}")

        max_cluster_stats_perm = []

        for perm_idx in range(n_permutations):
            # 打印进度
            if (perm_idx + 1) % 100 == 0:
                print(f"  ...置换进度: {perm_idx + 1}/{n_permutations}")

            # 在被试水平上置换组别标签
            permuted_groups = subject_info['Group'].sample(frac=1).values
            subject_info['Group_perm'] = permuted_groups
            df_perm = df_comp.merge(subject_info[['SubjectID', 'Group_perm']], on='SubjectID', how='left')
            df_perm.rename(columns={'Group': 'Group_orig', 'Group_perm': 'Group'}, inplace=True)

            t_perm = np.zeros(n_channels)
            for i, ch_num in enumerate(range(1, n_channels + 1)):
                ch_data_perm = df_perm[df_perm['Channel'] == ch_num]
                if not ch_data_perm.empty:
                    t_perm[i] = run_lme_on_channel(ch_data_perm, param, group_a, group_b)

            # 使用MNE的函数寻找置换数据中的簇
            clusters_perm, cluster_stats_perm = mne.stats.find_clusters(
                t_perm, threshold=t_threshold, connectivity=adjacency, tail=0  # tail=0 for two-tailed test
            )



            if len(cluster_stats_perm) > 0:
                max_cluster_stats_perm.append(np.max(cluster_stats_perm))
            else:
                max_cluster_stats_perm.append(0)

        # 4.4 比较真实簇与零分布
        print("\n步骤 3/3: 评估真实簇的显著性...")
        real_clusters, real_cluster_stats = mne.stats.find_clusters(
            t_obs, threshold=t_threshold, connectivity=adjacency, tail=0
        )

        significant_clusters_mask = np.zeros(n_channels, dtype=bool)
        if len(real_clusters) > 0:
            for i, cluster in enumerate(real_clusters):
                p_val = np.mean(np.array(max_cluster_stats_perm) >= real_cluster_stats[i])
                if p_val < montecarlo_alpha:
                    print(f"  >>> 发现显著簇! P = {p_val:.4f}, 包含电极: {[ch_names[i] for i in np.where(cluster)[0]]}")
                    significant_clusters_mask[cluster] = True

        if not np.any(significant_clusters_mask):
            print("  在此次比较中未发现任何显著的电极簇。")

        # 4.5 可视化
        fig, ax = plt.subplots(figsize=(6, 5))
        title = f'LME t-values: {param}\n({comp_name}, Age adjusted)'

        im, _ = mne.viz.plot_topomap(
            data=t_obs, pos=info, axes=ax, show=False, cmap='RdBu_r',
            mask=significant_clusters_mask,
            mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=5)
        )
        ax.set_title(title, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('t-value (LME)')
        plt.show()

print("\n" + "=" * 80)
print("所有分析完成！")