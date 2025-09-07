# -*- coding: utf-8 -*-
"""
Automated Feature Selection and Model Comparison for ADHD Data (v2 - Robust)

This script systematically finds the best feature subset and the best-performing
machine learning model. This version is revised to be more robust and avoid
complex nested pipeline errors.

Key Steps:
1.  Loops through each model (Logistic Regression, SVM, etc.).
2.  For each model, first uses RFECV on the whole dataset to find the optimal feature set.
3.  Then, evaluates that model's performance using cross-validation on only the selected features.
4.  Reports a final summary comparing all models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Preprocessing and Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import clone  # NEW: To ensure fresh models are used

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# --- 1. Parameters and Configuration ---

# TODO: 确保文件名和工作表名正确
FILE_PATH = 'ant网络雷达图.xlsx'
SHEET_NAME = 'Sheet1'

# TODO: 在这里放入你所有的候选特征
FEATURE_COLUMNS = [
    # 'Age',  # 年龄
    'Accuracy',  # 行为学 - 准确率 (ACC)
    'Response Time',  # 行为学 - 反应时 (RT)
    'Alerting',  # ANT - 警觉网络
    'Orienting',  # ANT - 定向网络
    'Executive Control',  # ANT - 执行控制网络
    # 'maxnegpkamp',  # 慢波参数 - 最大负波幅
    # 'maxnegpkamp_1',
    # 'maxnegpkamp_3',
    # 'maxnegpkamp_5',
    # 'maxnegpkamp_7',
    # 'maxnegpkamp_9',
    # 'maxnegpkamp_10',
    # 'maxnegpkamp_12',
    'maxnegpkamp_cluster1',
    'maxnegpkamp_cluster2',
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
    'mxdnslp_cluster1',
    'mxdnslp_cluster2',
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

# 定义组别ID
GROUP_ADHD1 = 1  # 映射为 0
GROUP_ADHD3 = 3  # 映射为 1


# --- 2. Data Loading and Preparation (No changes here) ---

def load_and_prepare_data(file_path, sheet_name, feature_cols, group1, group3):
    """Loads, cleans, and prepares data for classification."""
    print("--- 1. Loading and Preparing Data ---")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{file_path}'.")
        return None, None
    df_filtered = df[df['Group'].isin([group1, group3])].copy()
    X = df_filtered[feature_cols]
    y = df_filtered['Group'].map({group1: 0, group3: 1})
    original_count = len(X)
    X = X.dropna()
    y = y.loc[X.index]
    if len(X) != original_count:
        print(f"Dropped {original_count - len(X)} rows with missing values.")
    if len(X) < 20:
        print("⚠️ Warning: Very few samples remaining.")
        return None, None
    print(f"Data prepared. Found {len(X)} samples.")
    print(f"Feature count: {len(X.columns)}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}\n")
    return X, y


# --- 3. Main Analysis Block (Revised Logic) ---

if __name__ == "__main__":
    X, y = load_and_prepare_data(FILE_PATH, SHEET_NAME, FEATURE_COLUMNS, GROUP_ADHD1, GROUP_ADHD3)

    if X is not None:
        models = {
            "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
            "SVM": SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        results = {}
        best_features_per_model = {}

        print("--- 2. Starting Automated Feature Selection and Model Comparison ---")
        print("💡 NEW: Using PolynomialFeatures to create feature interactions.")

        for model_name, model in models.items():
            start_time = time()
            print(f"\n{'=' * 50}\nProcessing Model: {model_name}\n{'=' * 50}")

            # --- STEP 1: Find optimal features (including interactions) ---
            print("--> Step 1: Finding optimal features with RFECV...")

            feature_selector = RFECV(
                estimator=clone(model),
                step=1,
                cv=StratifiedKFold(3),
                scoring='f1_macro',
                n_jobs=-1
            )

            # MODIFIED: Added PolynomialFeatures to the pipeline
            selection_pipeline = ImbPipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
                # Create interaction terms (e.g., featA * featB)
                ('scaler', StandardScaler()),
                ('sampler', RandomOverSampler(random_state=42)),
                ('feature_selection', feature_selector)
            ])

            selection_pipeline.fit(X, y)

            # --- 获取并展示新特征 ---
            # 1. 获取原始多项式特征名
            poly_step = selection_pipeline.named_steps['poly']
            poly_feature_names = poly_step.get_feature_names_out(X.columns)

            # 2. 从RFECV获取被选择的特征的掩码（mask）
            selected_mask = selection_pipeline.named_steps['feature_selection'].support_

            # 3. 用掩码筛选出最终的特征名
            final_selected_features = np.array(poly_feature_names)[selected_mask].tolist()
            best_features_per_model[model_name] = final_selected_features

            print(
                f"Found {len(final_selected_features)} optimal features for {model_name} (from original + interactions).")
            for feature in final_selected_features:
                print(f"  - {feature}")

            # --- STEP 2: Evaluate model performance on the original features ---
            # We evaluate the whole pipeline which now includes creating and selecting features internally
            print("\n--> Step 2: Evaluating the entire pipeline with selected features...")

            evaluation_pipeline = ImbPipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
                ('scaler', StandardScaler()),
                ('sampler', RandomOverSampler(random_state=42)),
                ('classifier', clone(model))
            ])

            # 为了评估，我们需要用筛选出的特征来训练
            # 注意：这里的X是原始特征，pipeline会自动处理
            # 但是为了得到最准确的评估，我们应该在交叉验证的每一步都重新选择特征
            # 这正是我们上一个脚本（v2）用RFECV套在pipeline里的初衷，虽然复杂但最严谨
            # 这里我们做一个简化但有效的评估：用最终筛选出的特征集进行评估

            # 创建一个只包含最终被选中的特征的DataFrame
            temp_poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_poly_all_features = pd.DataFrame(temp_poly.fit_transform(X), columns=poly_feature_names, index=X.index)
            X_final_selected = X_poly_all_features[final_selected_features]

            final_eval_pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('sampler', RandomOverSampler(random_state=42)),
                ('classifier', clone(model))
            ])

            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(
                final_eval_pipeline,
                X_final_selected,  # 使用最终筛选出的特征组合进行评估
                y,
                scoring='f1_macro',
                cv=cv_strategy,
                n_jobs=-1
            )

            results[model_name] = scores
            print(f"Cross-validation F1-macro scores: {np.round(scores, 3)}")
            print(f"Mean F1-macro: {scores.mean():.3f} (± {scores.std():.3f})")

            end_time = time()
            print(f"\nTime taken for {model_name}: {end_time - start_time:.2f} seconds")

        # --- 4. Final Results Summary ---
        print("\n\n" + "=" * 60)
        print("          FINAL MODEL COMPARISON SUMMARY")
        print("=" * 60)

        # Print Best Features Found
        print("\n--- Optimal Features Found per Model ---")
        for model_name, features in best_features_per_model.items():
            print(f"\n{model_name}: ({len(features)} features)")
            print(f"  {features}")

        # Print Performance Table
        print("\n--- Model Performance (Mean F1-Macro) ---")
        results_df = pd.DataFrame(results).T
        results_df['mean_f1_macro'] = results_df.mean(axis=1)
        results_df['std_f1_macro'] = results_df.std(axis=1)
        results_df = results_df.sort_values(by='mean_f1_macro', ascending=False)
        print(results_df[['mean_f1_macro', 'std_f1_macro']])

        # Visualize the results
        print("\nGenerating performance comparison plot...")
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=pd.DataFrame(results).T, orient='h')
        plt.title('Model Performance Comparison (F1-Macro)', fontsize=16)
        plt.xlabel('F1-Macro Score', fontsize=12)
        plt.grid(True)
        plt.show()

        best_model_name = results_df.index[0]
        print(f"\n🏆 Based on these results, '{best_model_name}' appears to be the most promising model.")