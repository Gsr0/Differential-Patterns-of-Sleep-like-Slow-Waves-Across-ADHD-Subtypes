# -*- coding: utf-8 -*-
"""
SVM Classification Script for ADHD Slow Wave and Behavioral Data
This script performs SVM classification to distinguish between different participant groups
based on global slow wave parameters and behavioral data.

Tasks performed:
1. Load data from an Excel file.
2. Preprocess data for specific comparison groups.
3. Train and evaluate an SVM model for:
    a) Healthy Controls (HC) vs. All ADHD participants
    b) ADHD-I vs. ADHD-C subtypes
4. NEW: Perform feature selection to find the most effective features for subtype classification.
5. NEW: Calculate and visualize SHAP values to explain feature contributions.
6. Report key performance metrics including accuracy, precision, recall, F1-score,
   confusion matrix, and ROC curve.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
# NEW: Import RFE for feature selection and the shap library
from sklearn.feature_selection import RFE
import shap

# --- 1. Parameters and Configuration (参数设置) ---

# TODO: Fill in your file path and sheet name
# TODO: 请在这里填入你的Excel文件名和工作表名
FILE_PATH = 'ant网络雷达图.xlsx'  # <-- 修改这里
SHEET_NAME = 'Sheet1'  # <-- 修改这里 (如果需要)

# Define the feature columns to be used in the model
# 定义模型要使用的特征列
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
    'mxdnslp_cluster1',
    # 'mxdnslp_cluster2',
    # 'mxdnslp_cluster',
    # 'mxupslp_cluster',
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
# Define group IDs as they appear in your data file
# 定义数据文件中的组别ID
GROUP_HC = 0  # 健康儿童 (Healthy Controls)
GROUP_ADHD1 = 1  # ADHD-I 型
GROUP_ADHD3 = 3  # ADHD-C 型 (假设混合型是3, 请根据你的数据修改)


# --- 2. Data Loading and Preparation Functions (数据加载与准备函数) ---

def load_data(file_path, sheet_name):
    """Loads data from the specified Excel file."""
    # """从指定的Excel文件加载数据。"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✅ Data successfully loaded from '{file_path}'.")
        print(f"Total participants in file: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{file_path}'. Please check the file path.")
        return None


def prepare_data(df, features, target_groups, target_column='Group'):
    """
    Prepares data for a specific classification task by filtering groups
    and separating features (X) from the target (y).
    """
    # """为特定分类任务准备数据：筛选组别，并分离特征(X)和目标(y)。"""

    # Filter for the groups of interest
    df_filtered = df[df[target_column].isin(target_groups)].copy()

    # Handle missing values by dropping rows with any missing data in the required columns
    df_filtered.dropna(subset=features + [target_column], inplace=True)

    if len(df_filtered) < 10:  # Check if there's enough data
        print(f"⚠️ Warning: Not enough data for groups {target_groups}. Found only {len(df_filtered)} samples.")
        return None, None

    # Define features (X) and target (y)
    X = df_filtered[features]
    y = df_filtered[target_column]

    # Map group labels to binary 0 and 1 for SVM
    # 将组别标签映射为0和1，以供SVM使用
    y = y.map({target_groups[0]: 0, target_groups[1]: 1})

    print(f"Data prepared for groups {target_groups}.")
    print(f"Original features available: {len(X.columns)}")
    print(f"Number of samples: {len(X)}. Class distribution:\n{y.value_counts()}")

    return X, y


# --- 3. SVM Model Training and Evaluation Function (SVM模型训练与评估函数) ---

# MODIFIED: Function now includes feature selection and SHAP analysis capabilities
def run_svm_classification(X, y, model_name="SVM Classification", perform_feature_selection=False,
                           n_features_to_select=8):
    """
    Trains an SVM classifier, evaluates it, and visualizes the results.
    Optionally performs Recursive Feature Elimination (RFE) and SHAP analysis.
    """
    # """训练一个SVM分类器，进行评估和可视化。可选执行RFE特征选择和SHAP分析。"""

    # --- Train-Test Split (First Step) ---
    # Split data before any scaling or feature selection to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y  # Using 25% test size
    )

    # --- Feature Scaling (特征缩放) ---
    # Scale data based on the training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Keep track of the feature names being used
    current_feature_names = X.columns.tolist()

    # --- NEW: Feature Selection using RFE (特征选择) ---
    if perform_feature_selection:
        print("\n--- Performing Recursive Feature Elimination (RFE) ---")
        # The estimator used by RFE
        estimator = SVC(kernel='linear')
        # The RFE selector
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)

        # Fit RFE on the scaled training data
        selector = selector.fit(X_train_scaled, y_train)

        # Get the names of the selected features
        selected_mask = selector.support_
        current_feature_names = X.columns[selected_mask].tolist()

        print(f"Selected the top {len(current_feature_names)} features:")
        for feature in current_feature_names:
            print(f"- {feature}")

        # Reduce the datasets to only the selected features
        X_train_scaled = selector.transform(X_train_scaled)
        X_test_scaled = selector.transform(X_test_scaled)

    # --- Model Training (模型训练) ---
    # Initialize the final model. `probability=True` is needed for ROC curve.
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # --- Evaluation on Test Set (在测试集上评估) ---
    print("\n--- Test Set Evaluation Results ---")
    # MODIFIED: Added zero_division=0 to handle cases with no predicted samples for a class
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Visualization (结果可视化) ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)')
    ax.set_title(f'ROC Curve for {model_name}')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- NEW: SHAP Value Calculation and Visualization (SHAP值计算与可视化) ---
    print("\n--- Calculating and Visualizing SHAP Values ---")
    # SHAP works best with a background dataset (we use the training data)
    # For linear models, shap.LinearExplainer is efficient.
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    # Create the SHAP summary plot
    print("Generating SHAP summary plot...")
    plt.title(f"SHAP Feature Importance for {model_name}")
    shap.summary_plot(shap_values, X_test_scaled, feature_names=current_feature_names, show=False)
    plt.tight_layout()
    plt.show()


# --- 4. Main Execution Block (主执行模块) ---

if __name__ == "__main__":
    # Load the data first
    df_main = load_data(FILE_PATH, SHEET_NAME)

    if df_main is not None:
        # ===================================================================
        # Task 1: Healthy Controls (HC) vs. All ADHD Classification
        # ===================================================================
        print("\n" + "=" * 50)
        print("🚀 Task 1: Healthy Controls vs. All ADHD")
        print("=" * 50)

        df_task1 = df_main.copy()
        df_task1['ADHD_binary'] = df_task1['Group'].apply(lambda x: 0 if x == GROUP_HC else 1)
        X1, y1 = prepare_data(df_task1, FEATURE_COLUMNS, [0, 1], target_column='ADHD_binary')

        if X1 is not None and y1 is not None:
            # For this task, we use all features as it's already working well.
            run_svm_classification(X1, y1, model_name="HC vs. All ADHD", perform_feature_selection=False)
        else:
            print("Skipping Task 1 due to insufficient data.")

        # ===================================================================
        # Task 2: ADHD-I vs. ADHD-C Classification
        # ===================================================================
        print("\n" + "=" * 50)
        print("🚀 Task 2: ADHD-I vs. ADHD-C Subtype Classification")
        print("=" * 50)

        X2, y2 = prepare_data(df_main, FEATURE_COLUMNS, [GROUP_ADHD1, GROUP_ADHD3], target_column='Group')

        if X2 is not None and y2 is not None:
            # MODIFIED: Now we enable feature selection to find the best 8 features
            # and generate SHAP plots to understand their impact.
            # You can change n_features_to_select to any number you want to test.
            run_svm_classification(
                X2, y2,
                model_name="ADHD-I vs. ADHD-C",
                perform_feature_selection=True,
                n_features_to_select=8
            )
        else:
            print("Skipping Task 2 due to insufficient data.")

    print("\n✅ All tasks complete.")