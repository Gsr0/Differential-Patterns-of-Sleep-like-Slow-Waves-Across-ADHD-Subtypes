# -*- coding: utf-8 -*-
"""
优化版ADHD分类模型 - 特征工程与模型调优
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from scipy import stats
import shap

# --- 1. 参数配置 ---
FILE_PATH = 'ant网络雷达图.xlsx'
SHEET_NAME = 'Sheet1'
GROUP_HC = 0
GROUP_ADHD1 = 1
GROUP_ADHD3 = 3

# 所有可能特征（根据您的描述选择）
FEATURE_CANDIDATES = [
    # 'Age',  # 年龄
    # 'Accuracy',  # 行为学 - 准确率 (ACC)
    # 'Response Time',  # 行为学 - 反应时 (RT)
    # 'Alerting',  # ANT - 警觉网络
    # 'Orienting',  # ANT - 定向网络
    # 'Executive Control',  # ANT - 执行控制网络
    # 'maxnegpkamp',  # 慢波参数 - 最大负波幅
    # 'maxnegpkamp_1',
    # 'maxnegpkamp_3',
    # 'maxnegpkamp_5',
    # 'maxnegpkamp_7',
    # 'maxnegpkamp_9',
    # 'maxnegpkamp_10',
    # 'maxnegpkamp_12',
    'maxnegpkamp_cluster1',
    # 'maxnegpkamp_cluster2',
    # 'maxnegpkamp_cluster',
    # 'maxnegpkamp/mxdnslp_1',
    # 'mxdnslp_1',
    # 'mxdnslp_3',
    # 'mxdnslp_4',
    # 'mxdnslp_5',
    # 'mxdnslp_7',
    # 'mxdnslp_8',
    # 'mxdnslp_9',
    # 'mxdnslp_12',
    # 'mxdnslp_14',
    # 'mxdnslp_18',
    # 'mxdnslp_cluster1',
    # 'mxdnslp_cluster2',
    'mxdnslp_cluster',
    'mxupslp_cluster',
    # 'mxupslp_1',
    # 'mxupslp_3',
    # 'mxupslp_7',
    # 'mxupslp_5',
    # 'maxpospkamp',  # 慢波参数 - 最大正波幅
    # 'mxdnslp',  # 慢波参数 - 最大下降斜率
    # 'mxupslp',  # 慢波参数 - 最大上升斜率
    # 'sw_density',  # 慢波参数 - 慢波密度
    # 'mean_duration'  # 慢波参数 - 平均持续时间
]


# --- 2. 数据加载与预处理函数 ---
def load_and_preprocess_data(file_path, sheet_name):
    """加载并预处理数据"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✅ 数据成功加载: '{file_path}'")
        print(f"总样本数: {len(df)}")

        # 处理缺失值
        df.dropna(subset=FEATURE_CANDIDATES + ['Group'], inplace=True)

        # 创建ADHD二元标签
        df['ADHD_binary'] = df['Group'].apply(lambda x: 0 if x == GROUP_HC else 1)

        return df
    except Exception as e:
        print(f"❌ 加载数据错误: {e}")
        return None


# --- 3. 特征工程函数 ---
def feature_engineering(X, y):
    """执行特征工程"""
    # 1. 异常值处理 - Winsorization
    for col in X.columns:
        X[col] = stats.mstats.winsorize(X[col], limits=[0.05, 0.05])

    # 2. 特征变换 - 对数变换
    transformer = PowerTransformer(method='yeo-johnson')
    X_transformed = transformer.fit_transform(X)
    X = pd.DataFrame(X_transformed, columns=X.columns)

    return X, y


# --- 4. 特征选择函数 ---
def select_features(X, y):
    """使用多种方法选择最佳特征"""
    print("\n--- 特征选择 ---")

    # 方法1: 随机森林特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('随机森林特征重要性')
    plt.tight_layout()
    plt.show()

    # 方法2: RFECV (递归特征消除交叉验证)
    svm = SVC(kernel="linear", random_state=42)
    rfecv = RFECV(
        estimator=svm,
        step=1,
        cv=StratifiedKFold(5),
        scoring='accuracy',
        min_features_to_select=3
    )
    rfecv.fit(X, y)

    print(f"RFECV选择的最佳特征数: {rfecv.n_features_}")
    selected_features = X.columns[rfecv.support_]
    print(f"RFECV选择的特征: {list(selected_features)}")

    # 方法3: 相关性分析
    corr_matrix = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('特征相关性热图')
    plt.tight_layout()
    plt.show()

    # 综合选择特征 - 这里使用RFECV的结果
    return X[selected_features], selected_features


# --- 5. 模型训练与评估 ---
def train_and_evaluate(X, y, model_name="ADHD分类模型"):
    """训练并评估模型"""
    print(f"\n--- 训练模型: {model_name} ---")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建模型管道
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42)),  # 使用SMOTE处理不平衡
        ('classifier', SVC(probability=True, random_state=42))
    ])

    # 定义参数网格 - 包含多种模型
    param_grid = [
        {
            'classifier': [SVC(probability=True, random_state=42)],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__gamma': ['scale', 'auto', 0.1, 1],
            'classifier__class_weight': [None, 'balanced']
        },
        {
            'classifier': [xgb.XGBClassifier(
                # use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                tree_method='hist',
                device='cpu'
            )],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8, 1.0],
            'classifier__colsample_bytree': [0.8, 1.0]
        }
    ]

    # 网格搜索
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(5),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    print("正在进行网格搜索...")
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print("\n最佳参数:")
    print(grid_search.best_params_)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]  # 正类的概率

    print("\n测试集性能:")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.show()

    # ROC曲线
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC 分数: {roc_auc:.3f}")

    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f'{model_name} - ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.tight_layout()
    plt.show()

    # SHAP解释（如果是树模型）
    if 'XGBClassifier' in str(type(best_model.named_steps['classifier'])):
        print("\n生成SHAP解释...")
        explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
        X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
        shap_values = explainer.shap_values(X_test_scaled)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
        plt.title('SHAP特征重要性')
        plt.tight_layout()
        plt.show()

    return best_model, grid_search.best_score_


# --- 6. 主执行模块 ---
if __name__ == "__main__":
    # 加载数据
    df = load_and_preprocess_data(FILE_PATH, SHEET_NAME)

    if df is not None:
        # ===================================================================
        # 任务1: HC vs ADHD
        # ===================================================================
        print("\n" + "=" * 50)
        print("🚀 任务1: 健康对照组 vs ADHD患者")
        print("=" * 50)

        # 准备数据
        X1 = df[FEATURE_CANDIDATES]
        y1 = df['ADHD_binary']

        # 特征工程
        X1, y1 = feature_engineering(X1, y1)

        # 特征选择
        X1_selected, selected_features = select_features(X1, y1)

        # 训练模型
        model1, cv_score1 = train_and_evaluate(X1_selected, y1, "HC vs ADHD")

        # ===================================================================
        # 任务2: ADHD-I vs ADHD-C
        # ===================================================================
        print("\n" + "=" * 50)
        print("🚀 任务2: ADHD-I vs ADHD-C 亚型分类")
        print("=" * 50)

        # 筛选ADHD患者
        adhd_df = df[df['ADHD_binary'] == 1]

        # 准备数据
        X2 = adhd_df[FEATURE_CANDIDATES]
        y2 = adhd_df['Group'].map({GROUP_ADHD1: 0, GROUP_ADHD3: 1})

        # 特征工程
        X2, y2 = feature_engineering(X2, y2)

        # 特征选择
        X2_selected, _ = select_features(X2, y2)

        # 训练模型
        model2, cv_score2 = train_and_evaluate(X2_selected, y2, "ADHD-I vs ADHD-C")

        # 最终报告
        print("\n" + "=" * 50)
        print("📊 最终性能报告")
        print("=" * 50)
        print(f"任务1: HC vs ADHD - 交叉验证准确率: {cv_score1:.4f}")
        print(f"任务2: ADHD-I vs ADHD-C - 交叉验证准确率: {cv_score2:.4f}")

    print("\n✅ 所有任务完成")