"""
Customer Lifetime Value (CLV) Prediction: Two-Stage Pipeline
=============================================================
Stage 1: Classification  → Will the customer purchase in the next 90 days? (Yes/No)
Stage 2: Regression      → If yes, how much will they spend? (£)

Algorithms:
  Classification: Baseline (DummyClassifier), Logistic Regression, Random Forest, Gradient Boosting
  Regression: Baseline (Mean), Ridge, Random Forest, Gradient Boosting

Evaluation:
  Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
  Regression: MAE, RMSE, R2 (on active customers only → much cleaner signal)
"""
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# =============================================================================
# FEATURE ENGINEERING (Enhanced)
# =============================================================================
def prepare_feature_matrix(df: pd.DataFrame):
    """
    Builds an enriched feature matrix with 12 predictive features:
    - Original RFM: recency, frequency, monetary
    - Behavioral: tenure_days, aov, unique_products, total_items
    - Engineered: purchase_velocity, avg_items_per_order, recency_frequency_ratio,
                  monetary_per_day, product_diversity_ratio
    """
    df = df.copy()

    # ----- Engineered Features -----
    # Purchase velocity: orders per month of tenure
    df['purchase_velocity'] = df['frequency'] / (df['tenure_days'] / 30.0 + 1.0)

    # Average items per order
    df['avg_items_per_order'] = df['total_items'] / (df['frequency'] + 1.0)

    # Recency / Frequency ratio (high = dormant despite many past orders)
    df['recency_frequency_ratio'] = df['recency'] / (df['frequency'] + 1.0)

    # Monetary per day of tenure (spending intensity)
    df['monetary_per_day'] = df['monetary'] / (df['tenure_days'] + 1.0)

    # Product diversity ratio (unique products / total items)
    df['product_diversity_ratio'] = df['unique_products'] / (df['total_items'] + 1.0)

    feature_cols = [
        'recency', 'frequency', 'monetary',
        'tenure_days', 'aov', 'unique_products', 'total_items',
        'purchase_velocity', 'avg_items_per_order',
        'recency_frequency_ratio', 'monetary_per_day', 'product_diversity_ratio'
    ]

    X = df[feature_cols].copy()
    y_binary = df['purchased_next_90d'].copy()         # 0 or 1
    y_spend = df['target_spend_90d'].copy()             # continuous £

    # Handle infinite or NaN values
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    return X, y_binary, y_spend, feature_cols


# =============================================================================
# STAGE 1: CLASSIFICATION (Will customer purchase? Yes/No)
# =============================================================================
def train_classification_stage(X_train, y_train, X_test, y_test):
    """
    Trains 4 classifiers and evaluates Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    results = []
    trained_models = {}

    classifiers = {
        'Baseline (Stratified Dummy)': DummyClassifier(strategy='stratified', random_state=42),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        'Random Forest Classifier': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting Classifier': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            random_state=42
        )
    }

    print("\n" + "=" * 60)
    print("STAGE 1: CLASSIFICATION (Will Customer Purchase in Next 90 Days?)")
    print("=" * 60)

    for name, model in classifiers.items():
        print(f"\n--- Training: {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # For ROC-AUC, need probability scores
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('clf', model), 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.0

        results.append({
            'Model': name,
            'Accuracy': f'{acc * 100:.1f}%',
            'Precision': f'{prec * 100:.1f}%',
            'Recall': f'{rec * 100:.1f}%',
            'F1-Score': f'{f1 * 100:.1f}%',
            'ROC-AUC': f'{auc:.4f}',
            'acc_raw': acc
        })
        trained_models[name] = model

        print(f"  Accuracy:  {acc * 100:.1f}%")
        print(f"  Precision: {prec * 100:.1f}%")
        print(f"  Recall:    {rec * 100:.1f}%")
        print(f"  F1-Score:  {f1 * 100:.1f}%")
        print(f"  ROC-AUC:   {auc:.4f}")

    results_df = pd.DataFrame(results)
    print("\n--- Classification Benchmark Summary ---")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

    # Select best classifier by accuracy
    best_name = max(trained_models.keys(), key=lambda k: next(r['acc_raw'] for r in results if r['Model'] == k))
    best_clf = trained_models[best_name]
    print(f"\nBest Classifier: {best_name}")

    # Print confusion matrix for best model
    y_pred_best = best_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion Matrix ({best_name}):")
    print(f"  True Negatives:  {cm[0][0]}  |  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}  |  True Positives:  {cm[1][1]}")

    return results_df, trained_models, best_clf, best_name


# =============================================================================
# STAGE 1: TUNE BEST CLASSIFIER
# =============================================================================
def tune_best_classifier(X_train, y_train, best_name):
    """
    Hyperparameter tuning for the best classifier via 5-Fold Cross-Validation.
    """
    print("\n--- Tuning Best Classifier with 5-Fold Cross-Validation ---")

    if 'Gradient' in best_name:
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.05, 0.1, 0.15]
        }
        base_model = GradientBoostingClassifier(random_state=42)
    else:
        param_grid = {
            'n_estimators': [150, 200, 300],
            'max_depth': [8, 10, 12],
            'min_samples_split': [3, 5, 8]
        }
        base_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        base_model, param_grid, cv=cv,
        scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"Best CV Accuracy: {grid.best_score_ * 100:.1f}%")
    print(f"Best Hyperparameters: {grid.best_params_}")

    return grid.best_estimator_


# =============================================================================
# STAGE 2: REGRESSION (How much will active customers spend?)
# =============================================================================
def train_regression_stage(X_train, y_train, X_test, y_test):
    """
    Trains regressors ONLY on customers who actually purchased (non-zero target).
    This dramatically improves MAE and R2.
    """
    results = []
    trained_models = {}

    print("\n" + "=" * 60)
    print("STAGE 2: REGRESSION (How Much Will Active Customers Spend?)")
    print("=" * 60)
    print(f"Training on {len(X_train):,} active customers, testing on {len(X_test):,}")

    regressors = {
        'Baseline (Mean Dummy)': DummyRegressor(strategy='mean'),
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=10.0, random_state=42))
        ]),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting Regressor': HistGradientBoostingRegressor(
            max_iter=150, max_depth=8, learning_rate=0.05, random_state=42
        )
    }

    for name, model in regressors.items():
        print(f"\n--- Training: {name} ---")
        model.fit(X_train, y_train)
        y_pred = np.maximum(0, model.predict(X_test))

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
        trained_models[name] = model

        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2:   {r2:.4f}")

    results_df = pd.DataFrame(results)
    print("\n--- Regression Benchmark Summary ---")
    print(results_df.to_string(index=False))

    return results_df, trained_models


# =============================================================================
# STAGE 2: TUNE BEST REGRESSOR
# =============================================================================
def tune_best_regressor(X_train, y_train):
    """
    Tunes Gradient Boosting Regressor with 5-Fold CV on active customers only.
    """
    print("\n--- Tuning Gradient Boosting Regressor (5-Fold CV) ---")
    param_grid = {
        'max_iter': [100, 150, 200],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.1]
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_grid, cv=cv,
        scoring='neg_mean_absolute_error', n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"Best CV MAE: {-grid.best_score_:.2f}")
    print(f"Best Hyperparameters: {grid.best_params_}")

    return grid.best_estimator_


# =============================================================================
# FULL PIPELINE
# =============================================================================
def run_clv_pipeline(df: pd.DataFrame):
    os.makedirs(MODELS_DIR, exist_ok=True)

    X, y_binary, y_spend, feature_cols = prepare_feature_matrix(df)

    # Global train/test split (80/20)
    X_train, X_test, yb_train, yb_test, ys_train, ys_test = train_test_split(
        X, y_binary, y_spend, test_size=0.2, random_state=42, stratify=y_binary
    )
    print(f"\nGlobal Split: Train={len(X_train):,}, Test={len(X_test):,}")
    print(f"Class Distribution (Test): Active={yb_test.sum()} ({yb_test.mean()*100:.1f}%), Inactive={len(yb_test)-yb_test.sum()}")

    # =================== STAGE 1: CLASSIFICATION ===================
    clf_results, clf_models, best_clf, best_clf_name = train_classification_stage(
        X_train, yb_train, X_test, yb_test
    )

    # Tune best classifier
    tuned_clf = tune_best_classifier(X_train, yb_train, best_clf_name)

    # Evaluate tuned classifier on test set
    yb_pred_tuned = tuned_clf.predict(X_test)
    tuned_acc = accuracy_score(yb_test, yb_pred_tuned)
    tuned_f1 = f1_score(yb_test, yb_pred_tuned)
    tuned_proba = tuned_clf.predict_proba(X_test)[:, 1]
    tuned_auc = roc_auc_score(yb_test, tuned_proba)

    print(f"\n*** TUNED CLASSIFIER FINAL TEST RESULTS ***")
    print(f"  Accuracy:  {tuned_acc * 100:.1f}%")
    print(f"  F1-Score:  {tuned_f1 * 100:.1f}%")
    print(f"  ROC-AUC:   {tuned_auc:.4f}")

    cm = confusion_matrix(yb_test, yb_pred_tuned)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {cm[0][0]}  |  False Positives: {cm[0][1]}")
    print(f"    False Negatives: {cm[1][0]}  |  True Positives:  {cm[1][1]}")

    # =================== STAGE 2: REGRESSION (Active Customers Only) ===================
    # Filter to active customers only for regression training
    active_train_mask = yb_train == 1
    active_test_mask = yb_test == 1

    X_train_active = X_train[active_train_mask]
    ys_train_active = ys_train[active_train_mask]
    X_test_active = X_test[active_test_mask]
    ys_test_active = ys_test[active_test_mask]

    # Train regression directly on actual spend (no log transform for cleaner MAE)
    reg_results, reg_models = train_regression_stage(
        X_train_active, ys_train_active, X_test_active, ys_test_active
    )

    # Tune regressor on actual spend
    tuned_reg = tune_best_regressor(X_train_active, ys_train_active)

    # Evaluate tuned regressor on test set
    ys_pred_tuned = np.maximum(0, tuned_reg.predict(X_test_active))

    tuned_mae = mean_absolute_error(ys_test_active, ys_pred_tuned)
    tuned_rmse = root_mean_squared_error(ys_test_active, ys_pred_tuned)
    tuned_r2 = r2_score(ys_test_active, ys_pred_tuned)

    # Calculate baseline MAE for active customers only
    baseline_mae_active = mean_absolute_error(ys_test_active, np.full_like(ys_test_active, ys_train_active.mean()))
    improvement_pct = ((baseline_mae_active - tuned_mae) / baseline_mae_active) * 100

    print(f"\n*** TUNED REGRESSOR FINAL TEST RESULTS (Active Customers Only) ***")
    print(f"  MAE:  {tuned_mae:.2f} GBP")
    print(f"  RMSE: {tuned_rmse:.2f} GBP")
    print(f"  R2:   {tuned_r2:.4f}")
    print(f"  Baseline MAE (Mean): {baseline_mae_active:.2f} GBP")
    print(f"  MAE Improvement: {improvement_pct:.1f}%")

    # =================== COMBINED TWO-STAGE EVALUATION ===================
    # Simulate full pipeline: classify first, regress for predicted-active only
    yb_pred_test = tuned_clf.predict(X_test)
    combined_pred = np.zeros(len(X_test))
    predicted_active_mask = yb_pred_test == 1
    if predicted_active_mask.sum() > 0:
        combined_pred[predicted_active_mask] = np.maximum(0, tuned_reg.predict(X_test[predicted_active_mask]))

    combined_mae = mean_absolute_error(ys_test, combined_pred)
    combined_rmse = root_mean_squared_error(ys_test, combined_pred)
    combined_r2 = r2_score(ys_test, combined_pred)

    # Baseline for ALL customers (naive mean)
    baseline_mae_all = mean_absolute_error(ys_test, np.full_like(ys_test, ys_train.mean()))
    combined_improvement = ((baseline_mae_all - combined_mae) / baseline_mae_all) * 100

    print(f"\n*** COMBINED TWO-STAGE PIPELINE (ALL Customers) ***")
    print(f"  Combined MAE:  {combined_mae:.2f} GBP")
    print(f"  Combined RMSE: {combined_rmse:.2f} GBP")
    print(f"  Combined R2:   {combined_r2:.4f}")
    print(f"  Baseline MAE:  {baseline_mae_all:.2f} GBP")
    print(f"  Overall MAE Improvement: {combined_improvement:.1f}%")

    # =================== COMBINED TWO-STAGE PREDICTION ===================
    # For all customers: Stage1 predicts active/inactive, Stage2 predicts spend for predicted-active
    print("\n" + "=" * 60)
    print("COMBINED TWO-STAGE PIPELINE EVALUATION")
    print("=" * 60)

    # Full prediction on entire dataset
    X_full, _, y_spend_full, _ = prepare_feature_matrix(df)
    stage1_pred = tuned_clf.predict(X_full)
    stage1_proba = tuned_clf.predict_proba(X_full)[:, 1]

    # Stage 2 prediction only for predicted-active customers (no log transform)
    full_predictions = np.zeros(len(X_full))
    active_mask = stage1_pred == 1
    if active_mask.sum() > 0:
        full_predictions[active_mask] = np.maximum(0, tuned_reg.predict(X_full[active_mask]))

    # Save models
    joblib.dump(tuned_clf, os.path.join(MODELS_DIR, 'classifier_model.joblib'))
    joblib.dump(tuned_reg, os.path.join(MODELS_DIR, 'clv_best_model.joblib'))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.joblib'))

    # Save classification results
    clf_results.drop(columns=['acc_raw'], errors='ignore').to_csv(
        os.path.join(MODELS_DIR, 'classification_benchmark.csv'), index=False
    )

    # Save master dataset with predictions
    df_with_pred = df.copy()
    df_with_pred['predicted_spend_90d'] = full_predictions
    df_with_pred['purchase_probability'] = stage1_proba
    df_with_pred['predicted_active'] = stage1_pred
    df_with_pred.to_csv(os.path.join(DATA_DIR, 'customer_profiles_master.csv'), index=False)

    print(f"\nMaster customer profiles saved with two-stage predictions.")
    print(f"Models saved in: {MODELS_DIR}")

    # Summary metrics dict for the app
    metrics_summary = {
        'clf_accuracy': tuned_acc,
        'clf_f1': tuned_f1,
        'clf_auc': tuned_auc,
        'reg_mae': tuned_mae,
        'reg_rmse': tuned_rmse,
        'reg_r2': tuned_r2,
        'combined_mae': combined_mae,
        'combined_r2': combined_r2,
        'baseline_mae': baseline_mae_all,
        'mae_improvement_pct': combined_improvement,
        'confusion_matrix': cm.tolist()
    }
    joblib.dump(metrics_summary, os.path.join(MODELS_DIR, 'metrics_summary.joblib'))

    return tuned_clf, tuned_reg, clf_results


if __name__ == '__main__':
    customer_csv = os.path.join(DATA_DIR, 'customer_rfm_clv.csv')
    if os.path.exists(customer_csv):
        df_cust = pd.read_csv(customer_csv)
        run_clv_pipeline(df_cust)
    else:
        print(f"Run data_preprocessing.py first; {customer_csv} not found.")
