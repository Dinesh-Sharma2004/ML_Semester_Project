import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from time import time
import warnings

from dataset_generator import generate_synthetic_data

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, learning_curve
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.exceptions import DataConversionWarning, FitFailedWarning
from sklearn.utils.validation import DataConversionWarning as ValidationDataConversionWarning

numerical_cols = [
    'prev_gpa', 'avg_sleep_hours', 'sleep_variability',
    'self_reported_study_hours', 'library_days_per_week',
    'avg_daily_online_hours', 'typical_connection_window_start',
    'stress_level', 'mood_score', 'avg_steps_per_day'
]
categorical_cols = [
    'department', 'year', 'cafeteria_veg_pref',
    'roommate_conflicts_reported'
]
raw_features = numerical_cols + categorical_cols


def get_preprocessing_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor


def run_classification_experiment(X_train, y_train):
    print("\n--- Running Classification Experiment ---")
    preprocessor = get_preprocessing_pipeline()
    tscv = TimeSeriesSplit(n_splits=5)
    models_to_test = [
        {
            'name': 'RandomForest',
            'estimator': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'grid': {
                'model__n_estimators': [100, 150],
                'model__max_depth': [5, 10]
            }
        },
        {
            'name': 'LightGBM',
            'estimator': LGBMClassifier(random_state=42, is_unbalance=True),
            'grid': {
                'model__n_estimators': [100, 150],
                'model__learning_rate': [0.05, 0.1],
                'model__num_leaves': [20, 31]
            }
        }
    ]
    best_score = -1
    best_estimator = None

    for model_spec in models_to_test:
        print(f"\nTuning model: {model_spec['name']}...")
        start_time = time()
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model_spec['estimator'])
        ])
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_spec['grid'],
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"Model: {model_spec['name']} | Best AUC: {grid_search.best_score_:.4f} | Time: {time() - start_time:.1f}s")
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_estimator = grid_search.best_estimator_

    print(f"\n--- Experiment Complete ---")
    print(f"Winner: {best_estimator.named_steps['model'].__class__.__name__}")
    print(f"Best Tuned ROC-AUC (on Train CV): {best_score:.4f}")
    return best_estimator


def run_shap_analysis(pipeline, X_train):
    print("\n--- Running SHAP Analysis ---")
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    X_train_processed = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)

    if hasattr(model, 'feature_importances_'):
        print("Using TreeExplainer for tree-based model...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_processed_df)
    else:
        print("Using KernelExplainer for non-tree model (this may be slow)...")
        X_train_summary = shap.sample(X_train_processed_df, 100)
        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)
        print(f"Calculating KernelSHAP values for {len(X_train_processed_df)} training samples...")
        shap_values = explainer.shap_values(X_train_processed_df)

    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    elif len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    joblib.dump(explainer, 'shs_shap_explainer.joblib')
    X_train_processed_df.to_csv('shap_train_data.csv', index=False)
    np.save('shap_values_class1.npy', shap_values_class1)
    print("SHAP explainer and data saved successfully.")


def generate_and_save_learning_curve(pipeline, X, y):
    print("\n--- Generating Learning Curve Data ---")
    tscv = TimeSeriesSplit(n_splits=5)
    train_sizes = np.linspace(0.2, 1.0, 5)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ValidationDataConversionWarning)
        warnings.filterwarnings("ignore", category=DataConversionWarning)
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            X,
            y,
            cv=tscv,
            train_sizes=train_sizes,
            scoring='f1_weighted',
            n_jobs=-1,
            error_score=0
        )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    curve_data = {
        'train_sizes': train_sizes,
        'train_scores_mean': train_scores_mean,
        'train_scores_std': train_scores_std,
        'test_scores_mean': test_scores_mean,
        'test_scores_std': test_scores_std
    }
    joblib.dump(curve_data, 'learning_curve_data.joblib')
    print("Learning curve data saved.")


def save_artifacts(classifier_pipeline, regressor_pipeline, pca_model, df_dashboard):
    print("\n--- Saving Artifacts ---")
    joblib.dump(classifier_pipeline, 'shs_classifier_pipeline.joblib')
    joblib.dump(regressor_pipeline, 'shs_regressor_pipeline.joblib')
    joblib.dump(pca_model, 'shs_pca_model.joblib')
    df_dashboard.to_csv('dashboard_data.csv', index=False)


if __name__ == "__main__":
    full_df = generate_synthetic_data(n_students=500, n_weeks=10)
    df = full_df.drop_duplicates(subset=['student_id'], keep='last').reset_index(drop=True)
    print("\nData Schema:")
    print(df.info())

    X = df[raw_features]
    y_clf = df['at_risk_next_month']
    y_reg = df['semester_gpa']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_clf, y_test_clf = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
    y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

    best_classifier_pipeline = run_classification_experiment(X_train, y_train_clf)
    print("\n--- Final Model Evaluation on Test Set ---")
    y_pred_clf = best_classifier_pipeline.predict(X_test)
    y_pred_prob_clf = best_classifier_pipeline.predict_proba(X_test)[:, 1]
    print("Test Set Classification Report:")
    print(classification_report(y_test_clf, y_pred_clf))
    print(f"Test Set ROC-AUC: {roc_auc_score(y_test_clf, y_pred_prob_clf):.4f}")

    print("\nTraining simple regression model...")
    reg_preprocessor = get_preprocessing_pipeline()
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', reg_preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    reg_pipeline.fit(X_train, y_train_reg)

    run_shap_analysis(best_classifier_pipeline, X_train)
    generate_and_save_learning_curve(best_classifier_pipeline, X_train, y_train_clf)

    print("Preparing data for dashboard...")
    clf_preprocessor = best_classifier_pipeline.named_steps['preprocessor']
    X_processed_all = clf_preprocessor.transform(X)
    pca = PCA(n_components=3, random_state=42)
    pca.fit(X_processed_all[:split_idx])
    X_pca_all = pca.transform(X_processed_all)
    df_dashboard = df.copy()
    df_dashboard['PC1'] = X_pca_all[:, 0]
    df_dashboard['PC2'] = X_pca_all[:, 1]
    df_dashboard['PC3'] = X_pca_all[:, 2]
    df_dashboard['risk_probability'] = best_classifier_pipeline.predict_proba(X)[:, 1]
    df_dashboard['risk_prediction'] = best_classifier_pipeline.predict(X)
    save_artifacts(best_classifier_pipeline, reg_pipeline, pca, df_dashboard)
    print("\nPipeline complete. Artifacts saved.")
    print("To run the dashboard: streamlit run dashboard.py")
