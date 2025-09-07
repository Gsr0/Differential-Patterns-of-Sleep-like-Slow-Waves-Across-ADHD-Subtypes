# 调节效应分析示例代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf


# 假设数据结构
# df包含：maxnegpkamp (慢波指标), RT_variability (反应时变异性), ADHD_subtype (亚型)

# 方法1：使用statsmodels进行调节效应分析
def moderation_analysis(df):
    """
    调节效应分析：ADHD亚型调节慢波指标对行为表现的影响
    """

    # 1. 中心化连续变量（可选但推荐）
    df['maxnegpkamp_centered'] = df['maxnegpkamp'] - df['maxnegpkamp'].mean()

    # 2. 编码分类变量
    df['ADHD_subtype_coded'] = df['ADHD_subtype'].map({'ADHD-I': 0, 'ADHD-C': 1})

    # 3. 调节效应模型
    # Y = b0 + b1*X + b2*W + b3*(X*W) + covariates
    model = smf.ols('''RT_variability ~ maxnegpkamp_centered + ADHD_subtype_coded + 
                      maxnegpkamp_centered:ADHD_subtype_coded + Age''',
                    data=df).fit()

    print("调节效应分析结果:")
    print("=" * 50)
    print(model.summary())

    # 4. 提取关键结果
    interaction_coef = model.params['maxnegpkamp_centered:ADHD_subtype_coded']
    interaction_pval = model.pvalues['maxnegpkamp_centered:ADHD_subtype_coded']

    print(f"\n交互效应系数: {interaction_coef:.3f}")
    print(f"交互效应p值: {interaction_pval:.3f}")

    return model


# 方法2：简单斜率分析 (Simple Slopes Analysis)
def simple_slopes_analysis(df):
    """
    简单斜率分析：分别计算两个亚型中慢波指标对行为的影响
    """

    # 分组分析
    adhd_i_data = df[df['ADHD_subtype'] == 'ADHD-I']
    adhd_c_data = df[df['ADHD_subtype'] == 'ADHD-C']

    # ADHD-I组的回归
    model_i = smf.ols('RT_variability ~ maxnegpkamp + Age', data=adhd_i_data).fit()
    slope_i = model_i.params['maxnegpkamp']
    pval_i = model_i.pvalues['maxnegpkamp']

    # ADHD-C组的回归
    model_c = smf.ols('RT_variability ~ maxnegpkamp + Age', data=adhd_c_data).fit()
    slope_c = model_c.params['maxnegpkamp']
    pval_c = model_c.pvalues['maxnegpkamp']

    print("\n简单斜率分析结果:")
    print("=" * 50)
    print(f"ADHD-I组: 斜率 = {slope_i:.3f}, p = {pval_i:.3f}")
    print(f"ADHD-C组: 斜率 = {slope_c:.3f}, p = {pval_c:.3f}")

    return {'slope_i': slope_i, 'slope_c': slope_c, 'pval_i': pval_i, 'pval_c': pval_c}


# 方法3：可视化调节效应
def plot_moderation_effect(df):
    """
    绘制调节效应图
    """
    plt.figure(figsize=(10, 6))

    # 为每个亚型绘制散点图和回归线
    for subtype, color in zip(['ADHD-I', 'ADHD-C'], ['blue', 'red']):
        subset = df[df['ADHD_subtype'] == subtype]

        # 散点图
        plt.scatter(subset['maxnegpkamp'], subset['RT_variability'],
                    c=color, alpha=0.6, label=subtype)

        # 回归线
        z = np.polyfit(subset['maxnegpkamp'], subset['RT_variability'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(subset['maxnegpkamp'].min(),
                              subset['maxnegpkamp'].max(), 100)
        plt.plot(x_range, p(x_range), color=color, linewidth=2)

    plt.xlabel('慢波负峰幅度 (maxnegpkamp)')
    plt.ylabel('反应时变异性 (RT_variability)')
    plt.title('调节效应：ADHD亚型调节慢波指标对反应时变异性的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 方法4：效应量计算
def calculate_effect_sizes(df):
    """
    计算调节效应的效应量
    """
    # 使用层次回归计算R²变化

    # 第一步：只包含主效应
    model1 = smf.ols('RT_variability ~ maxnegpkamp + ADHD_subtype_coded + Age',
                     data=df).fit()
    r2_step1 = model1.rsquared

    # 第二步：加入交互效应
    model2 = smf.ols('''RT_variability ~ maxnegpkamp + ADHD_subtype_coded + 
                        maxnegpkamp:ADHD_subtype_coded + Age''',
                     data=df).fit()
    r2_step2 = model2.rsquared

    # R²变化
    delta_r2 = r2_step2 - r2_step1

    print(f"\n效应量分析:")
    print("=" * 50)
    print(f"主效应模型 R² = {r2_step1:.3f}")
    print(f"交互效应模型 R² = {r2_step2:.3f}")
    print(f"R²变化 (ΔR²) = {delta_r2:.3f}")

    return delta_r2


# 主函数
def run_moderation_analysis(df):
    """
    运行完整的调节效应分析
    """
    print("开始调节效应分析...")

    # 1. 调节效应分析
    model = moderation_analysis(df)

    # 2. 简单斜率分析
    slopes = simple_slopes_analysis(df)

    # 3. 可视化
    plot_moderation_effect(df)

    # 4. 效应量计算
    effect_size = calculate_effect_sizes(df)

    return model, slopes, effect_size

# 使用示例
# model, slopes, effect_size = run_moderation_analysis(df)