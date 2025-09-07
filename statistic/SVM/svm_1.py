import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class ADHDClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.svm_model = None
        self.pipeline = None
        self.feature_names = None
        self.feature_importance = None

    def load_and_prepare_data(self, data_path=None, data_df=None):
        """
        Load and prepare the slow wave feature data
        Expected columns: participant_id, group, age, electrode,
                         maxnegpkamp, maxpospkamp, maxdnslp, maxupslp,
                         sw_density, mean_duration
        """
        if data_df is not None:
            self.df = data_df.copy()
        else:
            # If loading from file
            self.df = pd.read_csv(data_path)

        # Feature engineering: aggregate electrode-wise features to participant level
        self.feature_cols = ['maxnegpkamp', 'maxpospkamp', 'maxdnslp',
                             'maxupslp', 'sw_density', 'mean_duration']

        # Create participant-level features by aggregating across electrodes
        participant_features = []

        for participant in self.df['participant_id'].unique():
            participant_data = self.df[self.df['participant_id'] == participant]

            # Basic demographics
            group = participant_data['group'].iloc[0]
            age = participant_data['age'].iloc[0]

            # Global statistics across all electrodes
            global_features = {}
            for feature in self.feature_cols:
                global_features[f'{feature}_mean'] = participant_data[feature].mean()
                global_features[f'{feature}_std'] = participant_data[feature].std()
                global_features[f'{feature}_max'] = participant_data[feature].max()
                global_features[f'{feature}_min'] = participant_data[feature].min()

            # Regional features (if electrode position info available)
            # Frontal: Fp1, Fp2, F3, F4, F7, F8, Fz
            frontal_electrodes = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
            if 'electrode' in participant_data.columns:
                frontal_data = participant_data[participant_data['electrode'].isin(frontal_electrodes)]
                if len(frontal_data) > 0:
                    for feature in self.feature_cols:
                        global_features[f'{feature}_frontal_mean'] = frontal_data[feature].mean()

                # Central: C3, C4, Cz, FCz
                central_electrodes = ['C3', 'C4', 'Cz', 'FCz']
                central_data = participant_data[participant_data['electrode'].isin(central_electrodes)]
                if len(central_data) > 0:
                    for feature in self.feature_cols:
                        global_features[f'{feature}_central_mean'] = central_data[feature].mean()

                # Parietal: P3, P4, Pz, CP1, CP2
                parietal_electrodes = ['P3', 'P4', 'Pz', 'CP1', 'CP2']
                parietal_data = participant_data[participant_data['electrode'].isin(parietal_electrodes)]
                if len(parietal_data) > 0:
                    for feature in self.feature_cols:
                        global_features[f'{feature}_parietal_mean'] = parietal_data[feature].mean()

            # Combine all features
            participant_row = {
                'participant_id': participant,
                'group': group,
                'age': age,
                **global_features
            }

            participant_features.append(participant_row)

        self.participant_df = pd.DataFrame(participant_features)

        # Handle missing values
        self.participant_df = self.participant_df.fillna(self.participant_df.mean())

        # Prepare features and labels
        feature_columns = [col for col in self.participant_df.columns
                           if col not in ['participant_id', 'group']]
        self.X = self.participant_df[feature_columns].values
        self.y = self.label_encoder.fit_transform(self.participant_df['group'].values)
        self.feature_names = feature_columns
        self.group_names = self.label_encoder.classes_

        print(f"Data prepared: {len(self.participant_df)} participants")
        print(f"Features: {len(feature_columns)}")
        print(f"Groups: {dict(zip(range(len(self.group_names)), self.group_names))}")

        return self.X, self.y

    def feature_selection(self, method='rfe', n_features=20):
        """
        Perform feature selection
        Methods: 'rfe' (Recursive Feature Elimination), 'kbest' (SelectKBest), 'pca'
        """
        if method == 'rfe':
            # Use RFE with SVM
            estimator = SVC(kernel='linear', random_state=42)
            self.feature_selector = RFE(estimator, n_features_to_select=n_features)
        elif method == 'kbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'pca':
            self.feature_selector = PCA(n_components=n_features)

        X_selected = self.feature_selector.fit_transform(self.X, self.y)

        # Get selected feature names (for RFE and KBest)
        if method in ['rfe', 'kbest']:
            if hasattr(self.feature_selector, 'get_support'):
                selected_mask = self.feature_selector.get_support()
                self.selected_features = [self.feature_names[i] for i, selected in enumerate(selected_mask) if selected]
            else:
                self.selected_features = [f'PC{i + 1}' for i in range(n_features)]
        else:
            self.selected_features = [f'PC{i + 1}' for i in range(n_features)]

        print(f"Feature selection completed: {X_selected.shape[1]} features selected")
        return X_selected

    def train_svm_model(self, X, y, optimize_hyperparameters=True):
        """
        Train SVM model with hyperparameter optimization
        """
        if optimize_hyperparameters:
            # Hyperparameter grid for SVM
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            }

            # Grid search with cross-validation
            svm_base = SVC(random_state=42, probability=True)
            self.svm_model = GridSearchCV(
                svm_base, param_grid, cv=5,
                scoring='accuracy', n_jobs=-1, verbose=1
            )
        else:
            # Default SVM parameters
            self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale',
                                 random_state=42, probability=True)

        # Train the model
        self.svm_model.fit(X, y)

        if optimize_hyperparameters:
            print(f"Best parameters: {self.svm_model.best_params_}")
            print(f"Best cross-validation score: {self.svm_model.best_score_:.3f}")

        return self.svm_model

    def create_pipeline(self, feature_selection_method='rfe', n_features=20):
        """
        Create a complete pipeline with preprocessing, feature selection, and classification
        """
        steps = [
            ('scaler', StandardScaler()),
        ]

        # Add feature selection step
        if feature_selection_method == 'rfe':
            steps.append(('feature_selection', RFE(SVC(kernel='linear'), n_features_to_select=n_features)))
        elif feature_selection_method == 'kbest':
            steps.append(('feature_selection', SelectKBest(score_func=f_classif, k=n_features)))
        elif feature_selection_method == 'pca':
            steps.append(('feature_selection', PCA(n_components=n_features)))

        # Add classifier
        steps.append(('classifier', SVC(kernel='rbf', C=10, gamma='scale',
                                        random_state=42, probability=True)))

        self.pipeline = Pipeline(steps)
        return self.pipeline

    def cross_validation_evaluation(self, X, y, cv_folds=5):
        """
        Perform cross-validation evaluation
        """
        if self.pipeline is None:
            self.create_pipeline()

        # Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Cross-validation scores
        cv_scores = cross_val_score(self.pipeline, X, y, cv=skf, scoring='accuracy')

        print(f"Cross-validation results ({cv_folds}-fold):")
        print(f"Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"Individual fold scores: {cv_scores}")

        return cv_scores

    def evaluate_model(self, X_test, y_test, X_train=None, y_train=None):
        """
        Evaluate the trained model
        """
        if X_train is not None and y_train is not None:
            train_pred = self.pipeline.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)
            print(f"Training Accuracy: {train_accuracy:.3f}")

        # Test predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)

        # Classification metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_accuracy:.3f}")

        # Detailed classification report
        target_names = [f"Group_{i}" for i in range(len(self.group_names))]
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        return y_pred, y_pred_proba, cm

    def plot_confusion_matrix(self, cm, group_names=None):
        """
        Plot confusion matrix
        """
        if group_names is None:
            group_names = self.group_names

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=group_names, yticklabels=group_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self, y_test, y_pred_proba):
        """
        Plot ROC curves for multiclass classification
        """
        # Convert to binary format for ROC analysis
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(len(self.group_names)))

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        for i in range(len(self.group_names)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, linewidth=2,
                     label=f'{self.group_names[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if self.pipeline is None:
            print("Model not trained yet!")
            return None

        # For SVM, we can use the coefficients for linear kernel
        # or permutation importance for non-linear kernels
        from sklearn.inspection import permutation_importance

        # This requires X and y to be available
        if hasattr(self, 'X') and hasattr(self, 'y'):
            perm_importance = permutation_importance(
                self.pipeline, self.X, self.y,
                n_repeats=10, random_state=42
            )

            # Create feature importance DataFrame
            if hasattr(self, 'selected_features'):
                feature_names = self.selected_features
            else:
                feature_names = self.feature_names

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)

            return importance_df
        else:
            print("Training data not available for feature importance calculation")
            return None

    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        """
        importance_df = self.get_feature_importance()

        if importance_df is not None:
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)

            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Permutation Importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

            return top_features
        else:
            return None


# Example usage and complete analysis pipeline
def run_complete_analysis():
    """
    Complete analysis pipeline example
    """
    # Initialize classifier
    classifier = ADHDClassifier()

    # Example data loading (replace with your actual data)
    # For demonstration, creating sample data structure
    """
    Expected data format:
    participant_id | group | age | electrode | maxnegpkamp | maxpospkamp | maxdnslp | maxupslp | sw_density | mean_duration
    """

    # Load and prepare data
    # X, y = classifier.load_and_prepare_data('your_data.csv')

    # For demonstration, let's create sample data
    np.random.seed(42)
    n_HC, n_ADHD_I, n_ADHD_C = 19, 44, 57
    n_electrodes = 32

    sample_data = []

    # Generate sample data for each group
    for group, n_participants in [('HC', n_HC), ('ADHD-I', n_ADHD_I), ('ADHD-C', n_ADHD_C)]:
        for participant in range(n_participants):
            for electrode in range(n_electrodes):
                # Simulate different patterns for each group
                if group == 'HC':
                    base_values = [50, 25, -15, 10, 0.5, 1.2]
                elif group == 'ADHD-I':
                    base_values = [65, 35, -20, 15, 0.7, 1.5]
                else:  # ADHD-C
                    base_values = [70, 40, -25, 18, 0.8, 1.6]

                # Add noise
                values = [base + np.random.normal(0, base * 0.2) for base in base_values]

                sample_data.append({
                    'participant_id': f'{group}_{participant:03d}',
                    'group': group,
                    'age': np.random.normal(8.5, 1.5),
                    'electrode': f'E{electrode:02d}',
                    'maxnegpkamp': values[0],
                    'maxpospkamp': values[1],
                    'maxdnslp': values[2],
                    'maxupslp': values[3],
                    'sw_density': values[4],
                    'mean_duration': values[5]
                })

    sample_df = pd.DataFrame(sample_data)

    # Load and prepare data
    X, y = classifier.load_and_prepare_data(data_df=sample_df)

    # Perform cross-validation
    cv_scores = classifier.cross_validation_evaluation(X, y, cv_folds=5)

    # Train final model with feature selection
    classifier.create_pipeline(feature_selection_method='rfe', n_features=15)
    classifier.pipeline.fit(X, y)

    # Feature importance analysis
    importance_df = classifier.plot_feature_importance(top_n=15)
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

    # Split data for final evaluation (if you want to hold out a test set)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train on training set
    classifier.pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred, y_pred_proba, cm = classifier.evaluate_model(X_test, y_test, X_train, y_train)

    # Plot results
    classifier.plot_confusion_matrix(cm)
    classifier.plot_roc_curves(y_test, y_pred_proba)

    return classifier, cv_scores, importance_df


# Run the analysis
if __name__ == "__main__":
    # classifier, cv_scores, importance_df = run_complete_analysis()
    # 1. 准备您的数据（CSV格式）
    # 需要包含列: participant_id, group, age, electrode, maxnegpkamp, maxpospkamp, maxdnslp, maxupslp, sw_density, mean_duration

    # 2. 运行分析
    classifier = ADHDClassifier()
    X, y = classifier.load_and_prepare_data('所有慢波参数分电极总表.csv')

    # 3. 交叉验证评估
    cv_scores = classifier.cross_validation_evaluation(X, y)

    # 4. 训练最终模型
    classifier.create_pipeline(feature_selection_method='rfe', n_features=15)
    classifier.pipeline.fit(X, y)

    # 5. 特征重要性分析
    importance_df = classifier.plot_feature_importance()