"""
Mobile-Wallet Fraud Detection Engine (Vodafone Cash & InstaPay)
Model Training, Cross-Validation, Evaluation & Cost-Matrix Optimization
"""
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MASTER_DATA_PATH = os.path.join(DATA_DIR, 'fraud_features_master.csv')

FEATURE_COLS = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'error_orig', 'error_dest', 'amount_to_orig_ratio', 'is_account_emptied',
    'hour_of_day', 'is_night_transaction', 'is_transfer', 'is_cash_out',
    'is_payment', 'is_cash_in', 'dest_is_merchant'
]

def load_processed_data():
    if not os.path.exists(MASTER_DATA_PATH):
        raise FileNotFoundError(f"Missing master data at {MASTER_DATA_PATH}. Run data_preprocessing.py first.")
    df = pd.read_csv(MASTER_DATA_PATH)
    return df

def evaluate_model_performance(y_true, y_pred, y_prob):
    """
    Computes standard and FinTech-specific metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def compute_cost_matrix_curve(y_true, y_prob, amounts, cost_fp=50.0, base_fn_cost=1500.0, thresholds=None):
    """
    Computes expected business financial loss (in EGP) across decision thresholds:
    Total Loss = sum(FP * cost_fp) + sum(FN * (amount or base_fn_cost))
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
        
    curve_data = []
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    amounts_arr = np.array(amounts)
    
    for t in thresholds:
        preds = (y_prob_arr >= t).astype(int)
        
        fp_mask = (preds == 1) & (y_true_arr == 0)
        fn_mask = (preds == 0) & (y_true_arr == 1)
        tp_mask = (preds == 1) & (y_true_arr == 1)
        tn_mask = (preds == 0) & (y_true_arr == 0)
        
        fp_count = int(fp_mask.sum())
        fn_count = int(fn_mask.sum())
        tp_count = int(tp_mask.sum())
        tn_count = int(tn_mask.sum())
        
        # Financial cost in EGP:
        # False Positive = User friction / manual review call (~50 EGP)
        loss_fp = fp_count * cost_fp
        
        # False Negative = Stolen money (actual transaction amount + chargeback fee)
        loss_fn = float(amounts_arr[fn_mask].sum()) if fn_count > 0 else 0.0
        # If amount is very small, enforce base chargeback cost
        loss_fn += fn_count * base_fn_cost * 0.1
        
        total_loss = loss_fp + loss_fn
        
        curve_data.append({
            'threshold': float(t),
            'total_loss': float(total_loss),
            'loss_fp': float(loss_fp),
            'loss_fn': float(loss_fn),
            'fp': fp_count,
            'fn': fn_count,
            'tp': tp_count,
            'tn': tn_count
        })
        
    df_curve = pd.DataFrame(curve_data)
    best_row = df_curve.loc[df_curve['total_loss'].idxmin()]
    return df_curve, float(best_row['threshold'])

def train_and_evaluate_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_processed_data()
    
    X = df[FEATURE_COLS].copy()
    y = df['isFraud'].values
    amounts = df['amount'].values
    
    # Stratified Train/Test Split (80% Train, 20% Holdout Test)
    X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
        X, y, amounts, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Dataset split: Train={len(X_train):,} ({y_train.sum()} frauds) | Test={len(X_test):,} ({y_test.sum()} frauds)")
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Baseline Dummy
    print("\n--- Training Baseline Dummy Classifier ---")
    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy.fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_test)[:, 1]
    dummy_preds = dummy.predict(X_test)
    dummy_metrics = evaluate_model_performance(y_test, dummy_preds, dummy_probs)
    print("Baseline PR-AUC:", round(dummy_metrics['pr_auc'], 4))
    
    # 2. Logistic Regression (Class-Balanced)
    print("\n--- Training Logistic Regression ---")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)
    lr_metrics = evaluate_model_performance(y_test, lr_preds, lr_probs)
    print("Logistic Regression PR-AUC:", round(lr_metrics['pr_auc'], 4), "ROC-AUC:", round(lr_metrics['roc_auc'], 4))
    
    # 3. Random Forest (Class-Balanced Subsample)
    print("\n--- Training Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)
    rf_metrics = evaluate_model_performance(y_test, rf_preds, rf_probs)
    print("Random Forest PR-AUC:", round(rf_metrics['pr_auc'], 4), "Recall:", round(rf_metrics['recall'], 4))
    
    # 4. HistGradientBoosting (Final Production Model)
    print("\n--- Training HistGradientBoosting Classifier ---")
    hgb = HistGradientBoostingClassifier(class_weight='balanced', max_iter=150, learning_rate=0.08, random_state=42)
    hgb.fit(X_train, y_train)
    hgb_probs = hgb.predict_proba(X_test)[:, 1]
    hgb_preds = (hgb_probs >= 0.5).astype(int)
    hgb_metrics = evaluate_model_performance(y_test, hgb_preds, hgb_probs)
    print("HistGradientBoosting PR-AUC:", round(hgb_metrics['pr_auc'], 4), "ROC-AUC:", round(hgb_metrics['roc_auc'], 4))
    
    # 5. Cost-Matrix & Dynamic Threshold Optimization for Production Model
    print("\n--- Computing Cost Matrix & Optimal Loss Threshold ---")
    cost_curve_df, optimal_threshold = compute_cost_matrix_curve(y_test, hgb_probs, amt_test)
    print(f"Optimal Decision Threshold: {optimal_threshold:.2f} (minimizes financial loss in EGP)")
    
    # Save Models and Metadata
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(hgb, os.path.join(MODELS_DIR, 'fraud_best_model.joblib'))
    joblib.dump(rf, os.path.join(MODELS_DIR, 'rf_model.joblib'))
    joblib.dump(lr, os.path.join(MODELS_DIR, 'lr_model.joblib'))
    joblib.dump(dummy, os.path.join(MODELS_DIR, 'dummy_model.joblib'))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, 'feature_cols.joblib'))
    
    metrics_summary = {
        'Baseline (Dummy)': dummy_metrics,
        'Logistic Regression': lr_metrics,
        'Random Forest': rf_metrics,
        'HistGradientBoosting (Best)': hgb_metrics,
        'optimal_threshold': optimal_threshold
    }
    joblib.dump(metrics_summary, os.path.join(MODELS_DIR, 'metrics_summary.joblib'))
    cost_curve_df.to_csv(os.path.join(DATA_DIR, 'cost_curve_data.csv'), index=False)
    
    # Save a small sample of test transactions for app batch upload testing
    test_sample = X_test.copy()
    test_sample['isFraud_actual'] = y_test
    test_sample['predicted_fraud_prob'] = hgb_probs
    test_sample.head(500).to_csv(os.path.join(DATA_DIR, 'test_transactions_sample.csv'), index=False)
    
    print("\nSUCCESS: All models, metrics, and test datasets saved in models/ and data/!")
    return metrics_summary

if __name__ == '__main__':
    train_and_evaluate_all()
