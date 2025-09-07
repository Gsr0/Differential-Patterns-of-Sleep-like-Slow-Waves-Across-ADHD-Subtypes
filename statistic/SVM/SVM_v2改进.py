# -*- coding: utf-8 -*-
"""
优化的SVM分类模型 - ADHD慢波和行为数据分析
Optimized SVM Classification Script for ADHD Slow Wave and Behavioral Data

优化策略:
1. 特征选择和工程化
2. 超参数调优
3. 类别不平衡处理
4. 集成学习
5. 更强的交叉验证策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# --- 1. 配置参数 ---
FILE_PATH = 'ant网络雷达图.xlsx'
SHEET_NAME = 'Sheet1'

# 扩展特征列表
FEATURE_COLUMNS = [
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
    # 'maxnegpkamp_cluster1',
    # 'maxnegpkamp_cluster2',
    'maxnegpkamp_cluster',
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

# 可选的额外特征（如果数据中存在）
OPTIONAL_FEATURES = [
    # 'Age',
    # 'Response Time',
    # 'Alerting',
    # 'Orienting',
    # 'Executive Control',
    # 'maxpospkamp',
    # 'mxupslp',
    # 'sw_density',
    # 'mean_duration'
]

GROUP_HC = 0
GROUP_ADHD1 = 1
GROUP_ADHD3 = 3


# --- 2. 数据加载函数 ---
def load_data(file_path, sheet_name):
    """加载数据并进行基本检查"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✅ 数据加载成功: '{file_path}'")
        print(f"总样本数: {len(df)}")
        print(f"列名: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 '{file_path}'")
        return None


def prepare_data_advanced(df, base_features, optional_features, target_groups, target_column='Group'):
    """高级数据准备：特征工程和选择"""

    # 筛选目标组别
    df_filtered = df[df[target_column].isin(target_groups)].copy()

    # 检查哪些可选特征在数据中存在
    available_features = base_features.copy()
    for feature in optional_features:
        if feature in df_filtered.columns:
            available_features.append(feature)
            print(f"✅ 添加特征: {feature}")

    # 删除缺失值
    df_filtered.dropna(subset=available_features + [target_column], inplace=True)

    if len(df_filtered) < 10:
        print(f"⚠️ 警告: 数据不足，仅有 {len(df_filtered)} 个样本")
        return None, None, None

    # 特征工程：创建新特征
    print("\n🔧 特征工程...")
    feature_df = df_filtered[available_features].copy()

    # 创建比率特征
    if 'maxnegpkamp_cluster1' in feature_df.columns and 'maxnegpkamp_cluster2' in feature_df.columns:
        feature_df['cluster_ratio'] = feature_df['maxnegpkamp_cluster1'] / (feature_df['maxnegpkamp_cluster2'] + 1e-8)

    if 'mxdnslp_cluster1' in feature_df.columns and 'mxdnslp_cluster2' in feature_df.columns:
        feature_df['slope_ratio'] = feature_df['mxdnslp_cluster1'] / (feature_df['mxdnslp_cluster2'] + 1e-8)

    # 创建交互特征
    if 'Accuracy' in feature_df.columns and 'maxnegpkamp' in feature_df.columns:
        feature_df['acc_amplitude_interaction'] = feature_df['Accuracy'] * feature_df['maxnegpkamp']

    # 目标变量
    y = df_filtered[target_column].map({target_groups[0]: 0, target_groups[1]: 1})

    print(f"最终特征数: {len(feature_df.columns)}")
    print(f"样本数: {len(feature_df)}")
    print(f"类别分布:\n{y.value_counts()}")

    return feature_df, y, list(feature_df.columns)


# --- 3. 优化的SVM模型训练函数 ---
def run_optimized_svm_classification(X, y, feature_names, model_name="Optimized SVM"):
    """运行优化的SVM分类"""

    print(f"\n🚀 开始优化 {model_name}")
    print("=" * 50)

    # 1. 特征选择
    print("1️⃣ 特征选择...")

    # 使用SelectKBest进行特征选择
    k_best = min(len(feature_names), max(3, len(feature_names) // 2))
    selector = SelectKBest(score_func=f_classif, k=k_best)
    X_selected = selector.fit_transform(X, y)

    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"选择的特征: {selected_features}")

    # 2. 处理类别不平衡
    print("2️⃣ 处理类别不平衡...")

    # 检查类别分布
    class_counts = pd.Series(y).value_counts()
    print(f"类别分布: {dict(class_counts)}")

    # 如果类别不平衡严重，使用SMOTE
    if min(class_counts) / max(class_counts) < 0.7:
        print("使用SMOTE进行过采样...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_selected, y)
        print(f"平衡后的类别分布: {pd.Series(y_balanced).value_counts().to_dict()}")
    else:
        X_balanced, y_balanced = X_selected, y

    # 3. 超参数调优
    print("3️⃣ 超参数调优...")

    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }

    # 使用网格搜索
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=cv_strategy,
        scoring='f1',
        n_jobs=-1
    )

    svm_grid.fit(X_balanced, y_balanced)

    print(f"最佳参数: {svm_grid.best_params_}")
    print(f"最佳交叉验证F1得分: {svm_grid.best_score_:.3f}")

    # 4. 集成学习
    print("4️⃣ 构建集成模型...")

    # 创建多个模型
    svm_best = svm_grid.best_estimator_
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)

    # 投票分类器
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_best),
            ('rf', rf_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )

    # 5. 模型评估
    print("5️⃣ 模型评估...")

    # 使用原始数据进行评估（避免过拟合）
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.25, random_state=42, stratify=y
    )

    # 缩放特征
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    ensemble.fit(X_train_scaled, y_train)

    # 预测
    y_pred = ensemble.predict(X_test_scaled)
    y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

    # 评估结果
    print("\n📊 集成模型评估结果:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # AUC得分
    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"\nAUC得分: {auc_score:.3f}")

    # 6. 单独评估SVM模型
    print("\n📊 优化SVM模型评估结果:")
    svm_pred = svm_best.predict(X_test_scaled)
    print(classification_report(y_test, svm_pred))

    # 7. 交叉验证评估
    print("\n📊 交叉验证结果:")
    cv_scores = cross_val_score(ensemble, X_selected, y, cv=cv_strategy, scoring='f1')
    print(f"集成模型F1得分: {np.mean(cv_scores):.3f} (± {np.std(cv_scores):.3f})")

    cv_scores_svm = cross_val_score(svm_best, X_selected, y, cv=cv_strategy, scoring='f1')
    print(f"SVM模型F1得分: {np.mean(cv_scores_svm):.3f} (± {np.std(cv_scores_svm):.3f})")

    # 8. 特征重要性分析
    print("\n📊 特征重要性分析:")
    if hasattr(ensemble.named_estimators_['rf'], 'feature_importances_'):
        feature_importance = ensemble.named_estimators_['rf'].feature_importances_
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        print(importance_df)

    return ensemble, svm_best, selected_features


# --- 4. 主执行函数 ---
def main():
    """主执行函数"""
    # 加载数据
    df_main = load_data(FILE_PATH, SHEET_NAME)

    if df_main is None:
        return

    # 任务1: HC vs All ADHD
    print("\n" + "=" * 60)
    print("🎯 任务1: 健康儿童 vs 所有ADHD")
    print("=" * 60)

    df_task1 = df_main.copy()
    df_task1['ADHD_binary'] = df_task1['Group'].apply(lambda x: 0 if x == GROUP_HC else 1)

    X1, y1, features1 = prepare_data_advanced(
        df_task1, FEATURE_COLUMNS, OPTIONAL_FEATURES, [0, 1], 'ADHD_binary'
    )

    if X1 is not None:
        ensemble1, svm1, selected_features1 = run_optimized_svm_classification(
            X1, y1, features1, "HC vs All ADHD"
        )

    # 任务2: ADHD-I vs ADHD-C
    print("\n" + "=" * 60)
    print("🎯 任务2: ADHD-I vs ADHD-C")
    print("=" * 60)

    X2, y2, features2 = prepare_data_advanced(
        df_main, FEATURE_COLUMNS, OPTIONAL_FEATURES, [GROUP_ADHD1, GROUP_ADHD3], 'Group'
    )

    if X2 is not None:
        ensemble2, svm2, selected_features2 = run_optimized_svm_classification(
            X2, y2, features2, "ADHD-I vs ADHD-C"
        )

    print("\n✅ 所有任务完成!")
    print("\n💡 优化建议:")
    print("1. 如果可能，收集更多数据样本")
    print("2. 考虑使用深度学习方法（如果数据量足够）")
    print("3. 进行更多特征工程，如时间序列特征")
    print("4. 考虑使用半监督学习方法")


if __name__ == "__main__":
    main()