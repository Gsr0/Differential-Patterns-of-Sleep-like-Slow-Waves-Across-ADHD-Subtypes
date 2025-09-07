import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 设置pandas显示选项，显示所有列和完整内容
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)        # 不限制显示宽度
pd.set_option('display.max_colwidth', None) # 不限制列宽度
pd.set_option('display.expand_frame_repr', False)  # 不换行显示
# --- 1. 数据加载与准备 ---

try:
    # 加载您的Excel文件
    df = pd.read_excel('ant网络雷达图.xlsx')
except FileNotFoundError:
    print("错误：请确保名为 'ant网络雷达图.xlsx' 的文件与此脚本在同一文件夹中。")
    exit()

# 为了方便，我们假设列名如下，请根据您的实际列名进行修改
# 假设: 'Group' 是分组, 'Age' 是年龄
# 'Executive_Control' 是执行控制分数 (因变量Y)
# 'maxnegpkamp' 是您最关心的慢波指标 (自变量X)
# 请务必将下面的 '列名' 替换为您的真实列名
# BEHAVIOR_Y = 'Accuracy'
# BEHAVIOR_Y = 'Response Time'  # 行为学 - 反应时 (RT)
# BEHAVIOR_Y = 'Alerting'  # ANT - 警觉网络
BEHAVIOR_Y = 'Orienting'  # ANT - 定向网络
# BEHAVIOR_Y = 'Executive Control'  # ANT - 执行控制网络

# BRAIN_X = 'maxnegpkamp'
# BRAIN_X = 'maxnegpkamp_Fp1'  #Executive Control+++
# BRAIN_X = 'maxnegpkamp_3'
# BRAIN_X = 'maxnegpkamp_5'
# BRAIN_X = 'maxnegpkamp_F4'  #Orienting
# BRAIN_X = 'maxnegpkamp_9' #Orienting
# BRAIN_X = 'maxnegpkamp_10'
# BRAIN_X = 'maxnegpkamp_12'
# BRAIN_X = 'maxnegpkamp_cluster1' #Orienting
# BRAIN_X = 'maxnegpkamp_cluster2'
# BRAIN_X = 'maxnegpkamp_cluster'#Orienting
# BRAIN_X = 'maxnegpkamp/mxdnslp_1'
# BRAIN_X = 'mxdnslp_1'
# BRAIN_X = 'mxdnslp_3'#Orienting
# BRAIN_X = 'mxdnslp_4'
# BRAIN_X = 'mxdnslp_5'
# BRAIN_X = 'mxdnslp_7'
# BRAIN_X = 'mxdnslp_8'
# BRAIN_X = 'mxdnslp_9'
# BRAIN_X = 'mxdnslp_12'  #ACC
# BRAIN_X = 'mxdnslp_14'
# BRAIN_X = 'mxdnslp_18'
# BRAIN_X = 'mxdnslp_cluster1'#Orienting
# BRAIN_X = 'mxdnslp_cluster2'#Orienting
# BRAIN_X = 'mxdnslp_cluster'#Orienting
# BRAIN_X = 'mxupslp_cluster'  #Executive Control
# BRAIN_X = 'mxupslp_1'  #Executive Control
# BRAIN_X = 'mxupslp_3'
# BRAIN_X = 'mxupslp_7'
# BRAIN_X = 'mxupslp_5'
BRAIN_X = 'maxpospkamp' # Executive Control  #Orienting
# BRAIN_X = 'mxdnslp'  # 慢波参数 - 最大下降斜率
# BRAIN_X = 'mxupslp'  # 慢波参数 - 最大上升斜率
# BRAIN_X = 'sw_density' # 慢波参数 - 慢波密度
# BRAIN_X = 'mean_duration'  # 慢波参数 - 平均持续时间


# --- 2. 筛选数据并创建调节变量和中心化变量 ---

# 筛选出ADHD-I (值为1) 和 ADHD-C (值为3) 的数据
df_subset = df[df['Group'].isin([1, 3])].copy()

# 创建调节变量 W (Moderator): Group_ADHD_C
# ADHD-I 组为 0, ADHD-C 组为 1。这让结果解释起来更直观
df_subset['Group_ADHD_C'] = df_subset['Group'].apply(lambda x: 1 if x == 3 else 0)

# 对连续的自变量和协变量进行中心化 (减去均值)
# 这能减少多重共线性，并使模型中的主效应更易于解释
df_subset[f'{BRAIN_X}_centered'] = df_subset[BRAIN_X] - df_subset[BRAIN_X].mean()
df_subset['Age_centered'] = df_subset['Age'] - df_subset['Age'].mean()


# --- 3. 运行调节效应模型 ---

print("--- 调节效应分析 ---")
# 使用 pingouin.linear_regression 来构建包含交互项的模型
# 公式: Y ~ X + W + X*W + Covariate
# Y = BEHAVIOR_Y, X = BRAIN_X_centered, W = Group_ADHD_C
model = pg.linear_regression(
    X=df_subset[[f'{BRAIN_X}_centered', 'Group_ADHD_C', 'Age_centered']],
    y=df_subset[BEHAVIOR_Y],
    add_intercept=True
)

# 手动添加交互项
interaction_term = df_subset[f'{BRAIN_X}_centered'] * df_subset['Group_ADHD_C']
X_with_interaction = sm.add_constant(pd.concat([
    df_subset[[f'{BRAIN_X}_centered', 'Group_ADHD_C', 'Age_centered']],
    interaction_term.rename('Interaction')
], axis=1))

model_with_interaction = sm.OLS(df_subset[BEHAVIOR_Y], X_with_interaction).fit()

print("模型结果 (交互项为 'Interaction'):")
print(model_with_interaction.summary())
print("\n" + "="*50 + "\n")

# 核心解读：请查看上面结果中 'Interaction' 这一行的 P>|t| (即p值)。
# 如果这个p值小于0.05，则说明存在显著的调节效应！


# --- 4. 简单斜率分析 (Simple Slopes / Post-Hoc Analysis) ---
# 既然调节效应显著，我们需要看看到底是怎样的关系
print("--- 简单斜率分析 (事后检验) ---")

# 分析ADHD-I组 (Group_ADHD_C = 0)
df_i = df_subset[df_subset['Group_ADHD_C'] == 0]
slope_i = pg.linear_regression(X=df_i[[f'{BRAIN_X}_centered', 'Age_centered']], y=df_i[BEHAVIOR_Y])
print("ADHD-I 组: 慢波对行为的影响")
print(slope_i.round(3))
print("\n")


# 分析ADHD-C组 (Group_ADHD_C = 1)
df_c = df_subset[df_subset['Group_ADHD_C'] == 1]
slope_c = pg.linear_regression(X=df_c[[f'{BRAIN_X}_centered', 'Age_centered']], y=df_c[BEHAVIOR_Y])
print("ADHD-C 组: 慢波对行为的影响")
print(slope_c.round(3))
print("\n" + "="*50 + "\n")


# --- 5. 结果可视化 ---
print("--- 正在生成调节效应可视化图... ---")

# 使用seaborn的lmplot函数，并将其返回的FacetGrid对象赋值给变量'g'
# 'g'现在包含了图的全部信息，包括图例
g = sns.lmplot(
    data=df_subset,
    x=BRAIN_X,
    y=BEHAVIOR_Y,
    hue='Group_ADHD_C',    # 用这个0/1变量来区分颜色
    ci=None,               # 不显示置信区间，让图像更清晰
    palette=['#2b6a99', '#f16c23'], # 为ADHD-I和ADHD-C指定不同颜色
    height=6,
    aspect=1.15
)

# 使用seaborn的新函数 sns.move_legend() 来移动图例到左上角
# 这是比手动创建新图例更稳健的方法
sns.move_legend(g, "upper right", fontsize=13)

# 现在我们可以修改已经移动到左上角的图例
# 1. 将图例的标题 ("Group_ADHD_C") 设置为空字符串，从而隐藏它
g.legend.set_title('')

# 2. 定义新的标签
new_labels = ['ADHD-I', 'ADHD-C']

# 3. 遍历图例中的文本对象并设置新标签
#    这会替换掉原来的 '0' 和 '1'
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)

# 设置X轴和Y轴的标签
plt.xlabel(f'Slow-wave Parameter ({BRAIN_X})', fontsize=19)
plt.ylabel(f'ANT network ({BEHAVIOR_Y})', fontsize=19)

# 显示最终的图像
plt.show()