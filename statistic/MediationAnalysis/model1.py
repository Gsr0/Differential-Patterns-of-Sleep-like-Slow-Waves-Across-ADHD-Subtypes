import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MediationAnalysis:
    def __init__(self, data, X_col, M_col, Y_col, covariates=None):
        """
        中介效应分析类

        Parameters:
        -----------
        data: DataFrame
            包含所有变量的数据框
        X_col: str
            自变量列名（如：'Group'，编码为0=对照组，1=ADHD组）
        M_col: str
            中介变量列名（如：'SlowWave_Rate'）
        Y_col: str
            因变量列名（如：'Omission_Rate'）
        covariates: list
            协变量列名列表（如：['Age', 'Gender']）
        """
        self.data = data.copy()
        self.X_col = X_col
        self.M_col = M_col
        self.Y_col = Y_col
        self.covariates = covariates or []

        # 移除缺失值
        cols_to_check = [X_col, M_col, Y_col] + self.covariates
        self.data = self.data.dropna(subset=cols_to_check)

        print(f"样本量: {len(self.data)}")
        print(f"自变量: {X_col}")
        print(f"中介变量: {M_col}")
        print(f"因变量: {Y_col}")
        if self.covariates:
            print(f"协变量: {self.covariates}")

    def run_mediation_analysis(self, bootstrap_samples=5000, alpha=0.05):
        """执行完整的中介效应分析"""

        print("\n" + "=" * 50)
        print("中介效应分析结果")
        print("=" * 50)

        # 1. 描述性统计
        self._descriptive_stats()

        # 2. 相关分析
        self._correlation_analysis()

        # 3. 路径分析
        paths = self._path_analysis()

        # 4. Bootstrap检验
        bootstrap_results = self._bootstrap_mediation(bootstrap_samples, alpha)

        # 5. 效应量计算
        effect_sizes = self._calculate_effect_sizes(paths)

        # 6. 结果可视化
        self._plot_results(paths, bootstrap_results)

        # 7. 生成结果报告
        self._generate_report(paths, bootstrap_results, effect_sizes)

        return {
            'paths': paths,
            'bootstrap': bootstrap_results,
            'effect_sizes': effect_sizes
        }

    def _descriptive_stats(self):
        """描述性统计"""
        print("\n1. 描述性统计")
        print("-" * 30)

        # 按组别统计
        if self.data[self.X_col].dtype in ['int64', 'float64']:
            # 如果是连续变量，按中位数分组
            median_x = self.data[self.X_col].median()
            groups = self.data[self.X_col] >= median_x
            group_labels = ['Low', 'High']
        else:
            # 如果是分类变量
            groups = self.data[self.X_col]
            group_labels = ['对照组', 'ADHD组']

        desc_stats = []
        for i, label in enumerate(group_labels):
            if isinstance(groups.iloc[0], bool):
                group_data = self.data[groups == (i == 1)]
            else:
                group_data = self.data[groups == i]

            stats_dict = {
                '组别': label,
                '样本量': len(group_data),
                f'{self.M_col}_均值': group_data[self.M_col].mean(),
                f'{self.M_col}_标准差': group_data[self.M_col].std(),
                f'{self.Y_col}_均值': group_data[self.Y_col].mean(),
                f'{self.Y_col}_标准差': group_data[self.Y_col].std()
            }
            desc_stats.append(stats_dict)

        desc_df = pd.DataFrame(desc_stats)
        print(desc_df.round(3))

    def _correlation_analysis(self):
        """相关分析"""
        print("\n2. 相关分析")
        print("-" * 30)

        vars_to_correlate = [self.X_col, self.M_col, self.Y_col]
        corr_matrix = self.data[vars_to_correlate].corr()

        print("相关系数矩阵:")
        print(corr_matrix.round(3))

        # 计算显著性
        n = len(self.data)
        for i in range(len(vars_to_correlate)):
            for j in range(i + 1, len(vars_to_correlate)):
                var1, var2 = vars_to_correlate[i], vars_to_correlate[j]
                r = corr_matrix.iloc[i, j]
                t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                print(f"{var1} vs {var2}: r={r:.3f}, p={p_value:.3f}")

    def _path_analysis(self):
        """路径分析"""
        print("\n3. 路径分析")
        print("-" * 30)

        X = self.data[self.X_col].values.reshape(-1, 1)
        M = self.data[self.M_col].values.reshape(-1, 1)
        Y = self.data[self.Y_col].values

        # 添加协变量
        if self.covariates:
            covs = self.data[self.covariates].values
            X_with_covs = np.hstack([X, covs])
            M_with_covs = np.hstack([M, covs])
            XM_with_covs = np.hstack([X, M, covs])
        else:
            X_with_covs = X
            M_with_covs = M
            XM_with_covs = np.hstack([X, M])

        paths = {}

        # 路径c: X → Y (总效应)
        model_c = LinearRegression().fit(X_with_covs, Y)
        paths['c'] = {
            'coef': model_c.coef_[0],
            'intercept': model_c.intercept_,
            'r2': model_c.score(X_with_covs, Y),
            'model': model_c
        }

        # 路径a: X → M
        model_a = LinearRegression().fit(X_with_covs, self.data[self.M_col].values)
        paths['a'] = {
            'coef': model_a.coef_[0],
            'intercept': model_a.intercept_,
            'r2': model_a.score(X_with_covs, self.data[self.M_col].values),
            'model': model_a
        }

        # 路径b和c': M → Y, X → Y (控制M)
        model_bc = LinearRegression().fit(XM_with_covs, Y)
        paths['b'] = {
            'coef': model_bc.coef_[1],  # M的系数
            'model': model_bc
        }
        paths['c_prime'] = {
            'coef': model_bc.coef_[0],  # X的系数（控制M后）
            'intercept': model_bc.intercept_,
            'r2': model_bc.score(XM_with_covs, Y),
            'model': model_bc
        }

        # 计算显著性
        n = len(self.data)
        for path_name, path_info in paths.items():
            if 'model' in path_info:
                # 计算标准误和t值
                if path_name == 'c':
                    y_pred = path_info['model'].predict(X_with_covs)
                    residuals = Y - y_pred
                elif path_name == 'a':
                    y_pred = path_info['model'].predict(X_with_covs)
                    residuals = self.data[self.M_col].values - y_pred
                else:  # b和c'
                    y_pred = path_info['model'].predict(XM_with_covs)
                    residuals = Y - y_pred

                mse = np.mean(residuals ** 2)
                se = np.sqrt(mse)
                t_stat = path_info['coef'] / (se / np.sqrt(n))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                path_info['se'] = se / np.sqrt(n)
                path_info['t_stat'] = t_stat
                path_info['p_value'] = p_value

        # 打印结果
        print(f"路径c (总效应): β={paths['c']['coef']:.4f}, p={paths['c']['p_value']:.3f}")
        print(f"路径a (X→M): β={paths['a']['coef']:.4f}, p={paths['a']['p_value']:.3f}")
        print(f"路径b (M→Y): β={paths['b']['coef']:.4f}")
        print(f"路径c' (直接效应): β={paths['c_prime']['coef']:.4f}")

        return paths

    def _bootstrap_mediation(self, n_bootstrap=5000, alpha=0.05):
        """Bootstrap检验间接效应"""
        print(f"\n4. Bootstrap检验 (n={n_bootstrap})")
        print("-" * 30)

        indirect_effects = []

        for i in range(n_bootstrap):
            # 重采样
            bootstrap_data = resample(self.data, random_state=i)

            X_boot = bootstrap_data[self.X_col].values.reshape(-1, 1)
            M_boot = bootstrap_data[self.M_col].values
            Y_boot = bootstrap_data[self.Y_col].values

            # 添加协变量
            if self.covariates:
                covs_boot = bootstrap_data[self.covariates].values
                X_with_covs_boot = np.hstack([X_boot, covs_boot])
                XM_with_covs_boot = np.hstack([X_boot, M_boot.reshape(-1, 1), covs_boot])
            else:
                X_with_covs_boot = X_boot
                XM_with_covs_boot = np.hstack([X_boot, M_boot.reshape(-1, 1)])

            # 计算路径a和b
            try:
                model_a_boot = LinearRegression().fit(X_with_covs_boot, M_boot)
                model_b_boot = LinearRegression().fit(XM_with_covs_boot, Y_boot)

                a_coef = model_a_boot.coef_[0]
                b_coef = model_b_boot.coef_[1]

                indirect_effect = a_coef * b_coef
                indirect_effects.append(indirect_effect)
            except:
                continue

        indirect_effects = np.array(indirect_effects)

        # 计算置信区间
        lower_ci = np.percentile(indirect_effects, (alpha / 2) * 100)
        upper_ci = np.percentile(indirect_effects, (1 - alpha / 2) * 100)

        # 计算点估计
        point_estimate = np.mean(indirect_effects)

        # 判断显著性
        is_significant = not (lower_ci <= 0 <= upper_ci)

        bootstrap_results = {
            'indirect_effects': indirect_effects,
            'point_estimate': point_estimate,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'is_significant': is_significant
        }

        print(f"间接效应点估计: {point_estimate:.4f}")
        print(f"{(1 - alpha) * 100}% 置信区间: [{lower_ci:.4f}, {upper_ci:.4f}]")
        print(f"中介效应显著性: {'显著' if is_significant else '不显著'}")

        return bootstrap_results

    def _calculate_effect_sizes(self, paths):
        """计算效应量"""
        print("\n5. 效应量分析")
        print("-" * 30)

        total_effect = paths['c']['coef']
        direct_effect = paths['c_prime']['coef']
        indirect_effect = paths['a']['coef'] * paths['b']['coef']

        # 中介效应比例
        if total_effect != 0:
            mediation_ratio = indirect_effect / total_effect
        else:
            mediation_ratio = 0

        # 标准化效应量
        X_std = self.data[self.X_col].std()
        M_std = self.data[self.M_col].std()
        Y_std = self.data[self.Y_col].std()

        standardized_indirect = (paths['a']['coef'] * X_std / M_std) * (paths['b']['coef'] * M_std / Y_std)

        effect_sizes = {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'mediation_ratio': mediation_ratio,
            'standardized_indirect': standardized_indirect
        }

        print(f"总效应: {total_effect:.4f}")
        print(f"直接效应: {direct_effect:.4f}")
        print(f"间接效应: {indirect_effect:.4f}")
        print(f"中介效应比例: {mediation_ratio:.1%}")
        print(f"标准化间接效应: {standardized_indirect:.4f}")

        return effect_sizes

    def _plot_results(self, paths, bootstrap_results):
        """结果可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 路径图
        ax1 = axes[0, 0]
        ax1.text(0.1, 0.8, self.X_col, fontsize=12, ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax1.text(0.9, 0.8, self.Y_col, fontsize=12, ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax1.text(0.5, 0.5, self.M_col, fontsize=12, ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        # 箭头和系数
        ax1.annotate('', xy=(0.45, 0.55), xytext=(0.15, 0.75),
                     arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax1.text(0.25, 0.65, f'a={paths["a"]["coef"]:.3f}', fontsize=10, color='blue')

        ax1.annotate('', xy=(0.85, 0.75), xytext=(0.55, 0.55),
                     arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        ax1.text(0.7, 0.6, f'b={paths["b"]["coef"]:.3f}', fontsize=10, color='green')

        ax1.annotate('', xy=(0.85, 0.85), xytext=(0.15, 0.85),
                     arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax1.text(0.5, 0.9, f"c'={paths['c_prime']['coef']:.3f}", fontsize=10, color='red')

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0.3, 1)
        ax1.set_title('中介效应路径图', fontsize=14)
        ax1.axis('off')

        # 2. Bootstrap分布
        ax2 = axes[0, 1]
        ax2.hist(bootstrap_results['indirect_effects'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(bootstrap_results['point_estimate'], color='red', linestyle='--', linewidth=2, label='点估计')
        ax2.axvline(bootstrap_results['lower_ci'], color='orange', linestyle='--', linewidth=2, label='置信区间')
        ax2.axvline(bootstrap_results['upper_ci'], color='orange', linestyle='--', linewidth=2)
        ax2.set_xlabel('间接效应')
        ax2.set_ylabel('频数')
        ax2.set_title('Bootstrap间接效应分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 散点图 X vs M
        ax3 = axes[1, 0]
        scatter = ax3.scatter(self.data[self.X_col], self.data[self.M_col],
                              c=self.data[self.Y_col], cmap='viridis', alpha=0.6)
        ax3.set_xlabel(self.X_col)
        ax3.set_ylabel(self.M_col)
        ax3.set_title('X vs M (颜色表示Y值)')
        plt.colorbar(scatter, ax=ax3, label=self.Y_col)

        # 4. 散点图 M vs Y
        ax4 = axes[1, 1]
        scatter2 = ax4.scatter(self.data[self.M_col], self.data[self.Y_col],
                               c=self.data[self.X_col], cmap='coolwarm', alpha=0.6)
        ax4.set_xlabel(self.M_col)
        ax4.set_ylabel(self.Y_col)
        ax4.set_title('M vs Y (颜色表示X值)')
        plt.colorbar(scatter2, ax=ax4, label=self.X_col)

        plt.tight_layout()
        plt.show()

    def _generate_report(self, paths, bootstrap_results, effect_sizes):
        """生成分析报告"""
        print("\n" + "=" * 50)
        print("中介效应分析报告")
        print("=" * 50)

        # 判断中介类型
        if bootstrap_results['is_significant']:
            if abs(paths['c_prime']['coef']) < 0.001 or paths['c_prime']['p_value'] > 0.05:
                mediation_type = "完全中介"
            else:
                mediation_type = "部分中介"
        else:
            mediation_type = "无中介效应"

        print(f"\n中介效应类型: {mediation_type}")
        print(f"中介效应大小: {effect_sizes['indirect_effect']:.4f}")
        print(f"中介效应比例: {effect_sizes['mediation_ratio']:.1%}")

        # 结论
        print(f"\n结论:")
        if bootstrap_results['is_significant']:
            print(f"- {self.X_col} 通过 {self.M_col} 对 {self.Y_col} 产生显著的间接效应")
            print(f"- 间接效应占总效应的 {effect_sizes['mediation_ratio']:.1%}")
            print(f"- 这支持了 {self.M_col} 作为中介变量的假设")
        else:
            print(f"- {self.M_col} 在 {self.X_col} 与 {self.Y_col} 之间未发现显著的中介效应")

        return {
            'mediation_type': mediation_type,
            'conclusion': f"{self.X_col} 通过 {self.M_col} 对 {self.Y_col} 的中介效应" +
                          ("显著" if bootstrap_results['is_significant'] else "不显著")
        }


def load_and_prepare_data(file_path):
    """加载和准备数据"""
    print("加载数据...")
    data = pd.read_excel(file_path)

    print("数据预览:")
    print(data.head())
    print(f"\n数据维度: {data.shape}")
    print(f"列名: {data.columns.tolist()}")

    return data


def run_multiple_mediation_models(data):
    """运行多个中介效应模型"""

    # 模型1: 漏判分析
    print("\n" + "=" * 60)
    print("模型1: ADHD亚型 → 漏判前慢波参数 → 漏判率")
    print("=" * 60)

    model1 = MediationAnalysis(
        data=data,
        X_col='Group',  # 0=对照组, 1=ADHD组
        M_col='SlowWave_Omission_Rate',  # 漏判前慢波发生率
        Y_col='Omission_Rate',  # 总漏判率
        covariates=['Age', 'Gender']  # 协变量
    )
    results1 = model1.run_mediation_analysis()

    # 模型2: 误判分析
    print("\n" + "=" * 60)
    print("模型2: ADHD亚型 → 误判前慢波参数 → 误判率")
    print("=" * 60)

    model2 = MediationAnalysis(
        data=data,
        X_col='Group',
        M_col='SlowWave_Commission_Rate',  # 误判前慢波发生率
        Y_col='Commission_Rate',  # 总误判率
        covariates=['Age', 'Gender']
    )
    results2 = model2.run_mediation_analysis()

    # 模型3: 执行网络分析
    print("\n" + "=" * 60)
    print("模型3: ADHD亚型 → 执行网络慢波活动 → 执行网络效率")
    print("=" * 60)

    model3 = MediationAnalysis(
        data=data,
        X_col='Group',
        M_col='SlowWave_Executive_Network',  # 执行网络相关慢波
        Y_col='Executive_Network_Efficiency',  # 执行网络效率
        covariates=['Age', 'Gender']
    )
    results3 = model3.run_mediation_analysis()

    # 模型4: 警觉网络分析
    print("\n" + "=" * 60)
    print("模型4: ADHD亚型 → 警觉网络慢波活动 → 警觉网络效率")
    print("=" * 60)

    model4 = MediationAnalysis(
        data=data,
        X_col='Group',
        M_col='SlowWave_Alerting_Network',  # 警觉网络相关慢波
        Y_col='Alerting_Network_Efficiency',  # 警觉网络效率
        covariates=['Age', 'Gender']
    )
    results4 = model4.run_mediation_analysis()

    return {
        'model1_omission': results1,
        'model2_commission': results2,
        'model3_executive': results3,
        'model4_alerting': results4
    }


# 主函数
if __name__ == "__main__":
    # 使用示例
    print("中介效应分析程序")
    print("请确保您的Excel文件包含以下列:")
    print("必需列:")
    print("- Group: 组别 (0=对照组, 1=ADHD组)")
    print("- Age: 年龄")
    print("- Gender: 性别")
    print("\n行为学相关列:")
    print("- Omission_Rate: 漏判率")
    print("- Commission_Rate: 误判率")
    print("- SlowWave_Omission_Rate: 漏判前慢波发生率")
    print("- SlowWave_Commission_Rate: 误判前慢波发生率")
    print("\nANT网络相关列:")
    print("- Executive_Network_Efficiency: 执行网络效率")
    print("- Alerting_Network_Efficiency: 警觉网络效率")
    print("- SlowWave_Executive_Network: 执行网络慢波参数")
    print("- SlowWave_Alerting_Network: 警觉网络慢波参数")

    # 加载数据
    # data = load_and_prepare_data('your_data.xlsx')

    # 运行分析
    # all_results = run_multiple_mediation_models(data)

    print("\n使用说明:")
    print("1. 将您的数据保存为Excel文件")
    print("2. 确保列名与上述要求一致")
    print("3. 取消注释最后两行代码")
    print("4. 修改文件路径为您的数据文件路径")
    print("5. 运行程序")