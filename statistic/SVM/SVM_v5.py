# -*- coding: utf-8 -*-
"""
优化版ADHD亚型分类模型 - 基于慢波特征和脑区特异性 (不使用skopt)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from scipy import stats
import shap
import joblib
from scipy.stats import uniform, randint

# --- 1. 参数配置 ---
FILE_PATH = 'ant网络雷达图.xlsx'
SHEET_NAME = 'Sheet1'
GROUP_HC = 0
GROUP_ADHD1 = 1
GROUP_ADHD3 = 3

# 所有可能特征（根据您的研究选择）
FRONTAL_ELECTRODES = [1, 3, 5]  # 额叶电极
ALL_FEATURES = [
    'maxnegpkamp', 'mxdnslp', 'mxupslp',
]

# 生成特征列名
FEATURE_COLUMNS = []
for feature in ALL_FEATURES:
    for electrode in FRONTAL_ELECTRODES:
        FEATURE_COLUMNS.append(f"{feature}_{electrode}")

print("使用的特征:", FEATURE_COLUMNS)


# --- 2. 数据加载与预处理函数 ---
def load_and_preprocess_data(file_path, sheet_name):
    """加载并预处理数据"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✅ 数据成功加载: '{file_path}'")
        print(f"总样本数: {len(df)}")

        # 处理缺失值
        df.dropna(subset=FEATURE_COLUMNS + ['Group'], inplace=True)

        # 创建ADHD二元标签
        df['ADHD_binary'] = df['Group'].apply(lambda x: 0 if x == GROUP_HC else 1)

        # 添加特征工程：创建额叶区域特征
        for feature in ALL_FEATURES:
            frontal_cols = [f"{feature}_{e}" for e in FRONTAL_ELECTRODES]
            df[f'frontal_{feature}_mean'] = df[frontal_cols].mean(axis=1)
            df[f'frontal_{feature}_std'] = df[frontal_cols].std(axis=1)
            FEATURE_COLUMNS.extend([f'frontal_{feature}_mean', f'frontal_{feature}_std'])

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

    # 3. 创建交互特征（额叶电极间的比值）
    for feature in ALL_FEATURES:
        for i in range(len(FRONTAL_ELECTRODES)):
            for j in range(i + 1, len(FRONTAL_ELECTRODES)):
                e1 = FRONTAL_ELECTRODES[i]
                e2 = FRONTAL_ELECTRODES[j]
                ratio_col = f"{feature}_ratio_{e1}_{e2}"
                X[ratio_col] = X[f"{feature}_{e1}"] / (X[f"{feature}_{e2}"] + 1e-6)

    return X, y


# --- 4. 特征选择函数 ---
def select_features(X, y):
    """使用多种方法选择最佳特征"""
    print("\n--- 特征选择 ---")

    # 方法1: 随机森林特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    plt.title('随机森林特征重要性')
    plt.tight_layout()
    plt.show()

    # 方法2: 互信息
    mi_scores = mutual_info_classif(X, y)
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values('MI_Score', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='MI_Score', y='Feature', data=mi_df.head(15))
    plt.title('互信息特征重要性')
    plt.tight_layout()
    plt.show()

    # 方法3: RFECV (递归特征消除交叉验证)
    svm = SVC(kernel="linear", random_state=42)
    rfecv = RFECV(
        estimator=svm,
        step=1,
        cv=StratifiedKFold(5),
        scoring='accuracy',
        min_features_to_select=5
    )
    rfecv.fit(X, y)

    print(f"RFECV选择的最佳特征数: {rfecv.n_features_}")
    selected_features = X.columns[rfecv.support_]
    print(f"RFECV选择的特征: {list(selected_features)}")

    # 可视化RFECV结果
    plt.figure()
    plt.xlabel("特征数量")
    plt.ylabel("交叉验证准确率")
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
    plt.title('RFECV结果')
    plt.tight_layout()
    plt.show()

    # 综合选择特征
    return X[selected_features], selected_features


# --- 5. 自定义评估指标 ---
def specificity_score(y_true, y_pred):
    """计算特异性（真阴性率）"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    return 0


# --- 6. 模型训练与评估 ---
def train_and_evaluate(X, y, model_name="ADHD亚型分类模型"):
    """训练并评估模型"""
    print(f"\n--- 训练模型: {model_name} ---")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建自定义评分器
    specificity_scorer = make_scorer(specificity_score)

    # 定义参数分布（使用RandomizedSearchCV替代BayesSearchCV）
    param_dist = [
        {
            'classifier': [SVC(probability=True, random_state=42)],
            'classifier__C': uniform(0.001, 1000),  # 对数均匀分布
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__gamma': ['scale', 'auto'] + list(uniform(0.0001, 10).rvs(10)),
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
            'classifier__n_estimators': randint(50, 300),
            'classifier__max_depth': randint(3, 10),
            'classifier__learning_rate': uniform(0.001, 0.3),  # 对数均匀分布
            'classifier__subsample': uniform(0.6, 0.4),  # 0.6-1.0
            'classifier__colsample_bytree': uniform(0.6, 0.4),  # 0.6-1.0
            'classifier__gamma': uniform(0, 5),
            'classifier__reg_alpha': uniform(0, 10),
            'classifier__reg_lambda': uniform(1, 9)  # 1-10
        }
    ]

    # 创建模型管道
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42)),  # 使用SMOTE处理不平衡
        ('classifier', SVC(probability=True, random_state=42))
    ])

    # 随机搜索优化
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=32,  # 迭代次数
        cv=StratifiedKFold(5),
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("正在进行随机搜索优化...")
    random_search.fit(X_train, y_train)

    # 输出最佳参数
    print("\n最佳参数:")
    print(random_search.best_params_)

    # 获取最佳模型
    best_model = random_search.best_estimator_

    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]  # 正类的概率

    print("\n测试集性能:")
    print(classification_report(y_test, y_pred))

    # 计算特异性
    specificity = specificity_score(y_test, y_pred)
    print(f"特异性: {specificity:.3f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ADHD-I', 'ADHD-C'],
                yticklabels=['ADHD-I', 'ADHD-C'])
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

    # SHAP解释
    try:
        if 'XGBClassifier' in str(type(best_model.named_steps['classifier'])):
            print("\n生成SHAP解释...")
            explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
            X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
            shap_values = explainer.shap_values(X_test_scaled)

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
            plt.title('SHAP特征重要性')
            plt.tight_layout()
            plt.show()

            # 单个样本解释
            plt.figure(figsize=(12, 6))
            shap.force_plot(explainer.expected_value, shap_values[0], X_test_scaled[0], feature_names=X.columns,
                            matplotlib=True)
            plt.title('单个样本SHAP解释')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"SHAP解释错误: {e}")

    # 保存模型
    joblib.dump(best_model, f'{model_name.replace(" ", "_")}_model.pkl')
    print(f"模型已保存为 '{model_name.replace(' ', '_')}_model.pkl'")

    return best_model, random_search.best_score_


# --- 7. 主执行模块 ---
if __name__ == "__main__":
    # 加载数据
    df = load_and_preprocess_data(FILE_PATH, SHEET_NAME)

    if df is not None:
        # ===================================================================
        # 任务2: ADHD-I vs ADHD-C
        # ===================================================================
        print("\n" + "=" * 50)
        print("🚀 任务2: ADHD-I vs ADHD-C 亚型分类")
        print("=" * 50)

        # 筛选ADHD患者
        adhd_df = df[df['ADHD_binary'] == 1]

        # 准备数据
        X = adhd_df[FEATURE_COLUMNS]
        y = adhd_df['Group'].map({GROUP_ADHD1: 0, GROUP_ADHD3: 1})

        # 检查样本分布
        print(f"ADHD-I样本数: {sum(y == 0)}")
        print(f"ADHD-C样本数: {sum(y == 1)}")

        # 特征工程
        X, y = feature_engineering(X, y)

        # 特征选择
        X_selected, selected_features = select_features(X, y)

        # 训练模型
        model, cv_score = train_and_evaluate(X_selected, y, "ADHD-I vs ADHD-C")

        # 最终报告
        print("\n" + "=" * 50)
        print("📊 最终性能报告")
        print("=" * 50)
        print(f"ADHD亚型分类交叉验证准确率: {cv_score:.4f}")

    print("\n✅ 所有任务完成")