# -*- coding: utf-8 -*-
"""
XGBoost Classification Script for ADHD Subtype Comparison (Fixed Version)

This script uses the powerful XGBoost algorithm to classify ADHD subtypes
and compares its performance with other models like SVM.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# 处理SHAP导入可能的错误
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not available. Feature importance analysis will be skipped.")
    SHAP_AVAILABLE = False

# 处理imblearn导入可能的错误
try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ imblearn not available. Using sklearn pipeline instead.")
    from sklearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = False

# --- 1. Parameters and Configuration ---
FILE_PATH = 'ant网络雷达图.xlsx'
SHEET_NAME = 'Sheet1'
FEATURE_COLUMNS = [
    # 'Age', 'Accuracy', 'Response Time', 'Alerting', 'Orienting',
    # 'Executive Control',
    'maxnegpkamp', 'maxnegpkamp_cluster1',
    'maxnegpkamp_cluster2', 'mxdnslp_cluster1', 'mxdnslp_cluster2',
    # 'maxpospkamp', 'mxdnslp', 'mxupslp', 'sw_density', 'mean_duration'
]
GROUP_ADHD1 = 1
GROUP_ADHD3 = 3


# --- 2. Data Loading and Preparation Functions ---
def load_data(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✅ Data successfully loaded from '{file_path}'.")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{file_path}'. Please check the file path.")
        return None
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return None


def prepare_data(df, features, target_groups, target_column='Group'):
    df_filtered = df[df[target_column].isin(target_groups)].copy()
    df_filtered.dropna(subset=features + [target_column], inplace=True)
    if len(df_filtered) < 10:
        print(f"⚠️ Warning: Not enough data for groups {target_groups}.")
        return None, None
    X = df_filtered[features]
    y = df_filtered[target_column].map({target_groups[0]: 0, target_groups[1]: 1})
    print(f"Data prepared for groups {target_groups}.")
    print(f"Number of samples: {len(X)}. Class distribution:\n{y.value_counts(normalize=True)}")
    return X, y


# --- 3. XGBoost Model Training and Evaluation Function ---
def run_xgboost_classification(X, y, model_name="XGBoost Classification"):
    """
    Trains and evaluates an XGBoost classifier with hyperparameter tuning.
    """
    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Create XGBoost Classifier with CPU-only configuration ---
    # 关键修复：禁用GPU，使用CPU模式，并设置更保守的参数
    xgb_classifier = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',  # 使用histogram方法，更稳定
        device='cpu',  # 强制使用CPU
        n_jobs=1,  # 使用单线程，避免并发问题
        verbosity=0  # 减少输出信息
    )

    # --- Create Pipeline ---
    if IMBLEARN_AVAILABLE:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('sampler', RandomOverSampler(random_state=42)),
            ('xgb', xgb_classifier)
        ])
    else:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb_classifier)
        ])

    # --- 简化的超参数网格，减少计算复杂度 ---
    print("\n--- Starting GridSearchCV for XGBoost ---")
    param_grid = {
        'xgb__n_estimators': [50, 100],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.1, 0.2],
        'xgb__subsample': [0.8, 1.0]
    }

    try:
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,  # 减少交叉验证折数
            scoring='f1_macro',
            n_jobs=1,  # 使用单线程
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        # --- Get the Best Model ---
        print("\n--- GridSearchCV Complete ---")
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation F1-macro score: {grid_search.best_score_:.3f}")

        best_model = grid_search.best_estimator_

    except Exception as e:
        print(f"❌ GridSearchCV failed: {e}")
        print("🔄 Using default parameters instead...")

        # 如果网格搜索失败，使用默认参数
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    # --- Predictions ---
    try:
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return

    # --- Evaluation on Test Set ---
    print("\n--- Test Set Evaluation Results (with XGBoost) ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # AUC Score
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC Score: {auc_score:.3f}")
    except Exception as e:
        print(f"⚠️ Could not calculate AUC: {e}")

    # --- Visualization ---
    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Could not create ROC curve: {e}")

    # --- Feature Importance (XGBoost built-in) ---
    try:
        print("\n--- Feature Importance Analysis ---")
        xgb_model = best_model.named_steps['xgb']
        feature_importance = xgb_model.feature_importances_

        # Create feature importance plot
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        print("Top 5 Most Important Features:")
        print(importance_df.head())

    except Exception as e:
        print(f"⚠️ Could not generate feature importance: {e}")

    # --- SHAP Value Analysis (if available) ---
    if SHAP_AVAILABLE:
        try:
            print("\n--- Calculating SHAP Values ---")
            # 获取XGBoost模型
            xgb_model = best_model.named_steps['xgb']

            # 准备数据
            if IMBLEARN_AVAILABLE:
                X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
                # 需要应用采样器来获得正确的特征
                X_test_resampled, _ = best_model.named_steps['sampler'].fit_resample(X_test_scaled, y_test)
                X_for_shap = X_test_resampled[:len(X_test)]  # 取原始测试集大小
            else:
                X_for_shap = best_model.named_steps['scaler'].transform(X_test)

            # 创建SHAP解释器
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_for_shap)

            # 生成SHAP摘要图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_for_shap, feature_names=X.columns, show=False)
            plt.title(f"SHAP Feature Importance for {model_name}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️ SHAP analysis failed: {e}")
    else:
        print("⚠️ SHAP not available for advanced feature analysis.")


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print("🚀 Starting XGBoost Classification Analysis")
    print("=" * 60)

    # Load data
    df_main = load_data(FILE_PATH, SHEET_NAME)
    if df_main is not None:
        print(f"\n📊 Dataset shape: {df_main.shape}")
        print(f"📋 Available columns: {list(df_main.columns)}")

        # Check if required columns exist
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df_main.columns]
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            print("Available columns in dataset:")
            print(df_main.columns.tolist())

            # Use only available columns
            available_features = [col for col in FEATURE_COLUMNS if col in df_main.columns]
            if available_features:
                print(f"🔄 Using available features: {available_features}")
                FEATURE_COLUMNS = available_features
            else:
                print("❌ No valid feature columns found.")
                exit(1)

        print("\n" + "=" * 60)
        print("🎯 Running XGBoost for: ADHD-I vs. ADHD-C")
        print("=" * 60)

        X, y = prepare_data(df_main, FEATURE_COLUMNS, [GROUP_ADHD1, GROUP_ADHD3], target_column='Group')
        if X is not None and y is not None:
            run_xgboost_classification(X, y)
        else:
            print("❌ Skipping analysis due to insufficient data.")
    else:
        print("❌ Failed to load data. Please check the file path and format.")

    print("\n✅ All tasks complete.")