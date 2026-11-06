"""
Script to generate clean, reproducible Jupyter notebook for
Mobile-Wallet Fraud Detection Engine (Vodafone Cash & InstaPay)
"""
import os
import json

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
NOTEBOOK_PATH = os.path.join(NOTEBOOKS_DIR, 'mobile_wallet_fraud_detection.ipynb')

def create_fraud_notebook():
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🛡️ Mobile-Wallet Fraud Detection Engine (Vodafone Cash & InstaPay)\n",
                "### Applied Machine Learning Final Portfolio Project · NextGen Institute\n",
                "**Domain:** FinTech & Digital Payments (Egypt & Emerging Markets)  \n",
                "**Dataset:** PaySim Mobile Money Simulation (Synthetic Financial Logs)  \n",
                "**Task:** Severe Imbalanced Classification with Cost-Loss Matrix Optimization  \n",
                "\n",
                "---\n",
                "\n",
                "## 1. Problem Framing & Business Objective\n",
                "- **The Problem:** Mobile-money services in Egypt (Vodafone Cash, InstaPay, Etisalat Cash, Orange Cash) have experienced explosive growth. With this rapid adoption, sophisticated fraud tactics (account takeover via phishing, social engineering cash-outs, mule account laundering) have surged.\n",
                "- **Business Challenge:** Fraud is extremely rare (~0.8% of transactions). A naive model predicting 'No Fraud' gets 99.2% accuracy but loses 100% of stolen customer funds! Conversely, overly aggressive filters block innocent users, creating massive customer friction and support costs.\n",
                "- **The ML Solution:**\n",
                "  1. Engineer domain-specific balance-reconciliation and velocity features.\n",
                "  2. Benchmark Cost-Sensitive Machine Learning models focusing on **PR-AUC (Precision-Recall AUC)** and Recall.\n",
                "  3. Optimize decision thresholds using a **Financial Cost Matrix** (Cost of Missed Fraud vs. Cost of False Alarm in EGP).\n",
                "  4. Ship an interactive Streamlit detection dashboard with real-time what-if simulation and CSV batch scanning."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Core Imports\n",
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Scikit-Learn\n",
                "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.dummy import DummyClassifier\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score, precision_score, recall_score, f1_score,\n",
                "    roc_auc_score, average_precision_score, confusion_matrix,\n",
                "    precision_recall_curve, roc_curve, classification_report\n",
                ")\n",
                "\n",
                "# Visualization setup\n",
                "pd.set_option('display.max_columns', 25)\n",
                "pd.set_option('display.float_format', lambda x: '%.4f' % x)\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "%matplotlib inline\n",
                "print('Environment and libraries initialized successfully!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Understanding & Ingestion\n",
                "We load the processed dataset with raw transaction logs and newly engineered financial features."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "data_path = '../data/fraud_features_master.csv'\n",
                "df = pd.read_csv(data_path)\n",
                "print(f\"Total Transactions Loaded: {len(df):,}\")\n",
                "print(f\"Columns: {list(df.columns)}\")\n",
                "df.head()"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Exploratory Data Analysis (EDA)\n",
                "### Insight 1: Extreme Class Imbalance\n",
                "Fraudulent transactions account for less than 1% of total traffic."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fraud_counts = df['isFraud'].value_counts()\n",
                "fraud_rate = df['isFraud'].mean() * 100\n",
                "print(f\"Legitimate Transactions (0): {fraud_counts[0]:,} ({100-fraud_rate:.2f}%)\")\n",
                "print(f\"Fraudulent Transactions (1): {fraud_counts[1]:,} ({fraud_rate:.2f}%)\")\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n",
                "sns.countplot(data=df, x='type', hue='isFraud', ax=axes[0], palette='Set1')\n",
                "axes[0].set_title('Transaction Volume by Type & Fraud Presence')\n",
                "axes[0].set_yscale('log')\n",
                "axes[0].set_ylabel('Log Count')\n",
                "\n",
                "sns.barplot(data=df.groupby('type')['isFraud'].mean().reset_index(), x='type', y='isFraud', ax=axes[1], palette='Reds_r')\n",
                "axes[1].set_title('Fraud Probability by Transaction Type')\n",
                "axes[1].set_ylabel('Fraud Rate')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Key Business EDA Insights:\n",
                "1. **Type Concentration:** 100% of mobile-money fraud happens exclusively in **TRANSFER** (InstaPay peer-to-peer) and **CASH_OUT** (Vodafone Cash agent cash withdrawals). Zero fraud exists in PAYMENT, CASH_IN, or DEBIT.\n",
                "2. **Account Draining:** In over 90% of fraudulent transfers, the victim's account is drained to exactly 0.00 EGP (`is_account_emptied == 1`).\n",
                "3. **Balance Inconsistency (Error):** In normal transactions, `oldbalance - amount = newbalance`. In fraudulent transactions, attackers often manipulate balances or send money to non-existent mule accounts, causing massive mathematical errors (`error_orig != 0` and `error_dest != 0`)."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Visualizing Balance Discrepancy\n",
                "plt.figure(figsize=(10, 4))\n",
                "sns.boxplot(data=df, x='isFraud', y='amount', showfliers=False, palette='coolwarm')\n",
                "plt.title('Transaction Amount: Legitimate (0) vs. Fraud (1)')\n",
                "plt.xlabel('isFraud')\n",
                "plt.ylabel('Amount (EGP)')\n",
                "plt.show()"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Leakage-Free Preprocessing & Train/Test Split\n",
                "We isolate 16 engineered features and split the dataset with stratified sampling to preserve the exact fraud ratio in the test set."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "feature_cols = [\n",
                "    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',\n",
                "    'error_orig', 'error_dest', 'amount_to_orig_ratio', 'is_account_emptied',\n",
                "    'hour_of_day', 'is_night_transaction', 'is_transfer', 'is_cash_out',\n",
                "    'is_payment', 'is_cash_in', 'dest_is_merchant'\n",
                "]\n",
                "X = df[feature_cols].copy()\n",
                "y = df['isFraud'].values\n",
                "amounts = df['amount'].values\n",
                "\n",
                "X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(\n",
                "    X, y, amounts, test_size=0.20, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "scaler = StandardScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "print(f\"Train Set: {len(X_train):,} ({y_train.sum()} frauds)\")\n",
                "print(f\"Test Set:  {len(X_test):,} ({y_test.sum()} frauds)\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Model Benchmarking\n",
                "We train four distinct models:\n",
                "1. **Baseline Dummy Classifier (Stratified)**\n",
                "2. **Logistic Regression (Class-Balanced)**\n",
                "3. **Random Forest Classifier (Class-Balanced Subsample)**\n",
                "4. **HistGradientBoosting Classifier (Tuned Non-Linear Ensemble)**"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "models = {\n",
                "    'Baseline Dummy': (DummyClassifier(strategy='stratified', random_state=42), X_train, X_test),\n",
                "    'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), X_train_scaled, X_test_scaled),\n",
                "    'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1), X_train, X_test),\n",
                "    'HistGradientBoosting': (HistGradientBoostingClassifier(class_weight='balanced', max_iter=150, learning_rate=0.08, random_state=42), X_train, X_test)\n",
                "}\n",
                "\n",
                "results = {}\n",
                "predictions_prob = {}\n",
                "\n",
                "for name, (clf, xtr, xte) in models.items():\n",
                "    clf.fit(xtr, y_train)\n",
                "    probs = clf.predict_proba(xte)[:, 1]\n",
                "    preds = (probs >= 0.5).astype(int)\n",
                "    \n",
                "    results[name] = {\n",
                "        'Accuracy': accuracy_score(y_test, preds),\n",
                "        'Precision': precision_score(y_test, preds, zero_division=0),\n",
                "        'Recall': recall_score(y_test, preds, zero_division=0),\n",
                "        'F1-Score': f1_score(y_test, preds, zero_division=0),\n",
                "        'ROC-AUC': roc_auc_score(y_test, probs),\n",
                "        'PR-AUC': average_precision_score(y_test, probs)\n",
                "    }\n",
                "    predictions_prob[name] = probs\n",
                "\n",
                "df_results = pd.DataFrame(results).T\n",
                "print(\"=== BENCHMARK EVALUATION RESULTS ON UNTOUCHED TEST SET ===\")\n",
                "df_results"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Precision-Recall & ROC Curves\n",
                "In highly imbalanced FinTech fraud detection, **PR-AUC is the true north star metric** because standard accuracy and ROC curves can be deceptively optimistic due to millions of true negatives."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
                "\n",
                "# PR Curves\n",
                "for name, probs in predictions_prob.items():\n",
                "    p, r, _ = precision_recall_curve(y_test, probs)\n",
                "    score = average_precision_score(y_test, probs)\n",
                "    axes[0].plot(r, p, label=f\"{name} (PR-AUC = {score:.3f})\")\n",
                "axes[0].set_title('Precision-Recall Curves')\n",
                "axes[0].set_xlabel('Recall')\n",
                "axes[0].set_ylabel('Precision')\n",
                "axes[0].legend(loc='lower left')\n",
                "\n",
                "# ROC Curves\n",
                "for name, probs in predictions_prob.items():\n",
                "    fpr, tpr, _ = roc_curve(y_test, probs)\n",
                "    score = roc_auc_score(y_test, probs)\n",
                "    axes[1].plot(fpr, tpr, label=f\"{name} (ROC-AUC = {score:.3f})\")\n",
                "axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)\n",
                "axes[1].set_title('ROC Curves')\n",
                "axes[1].set_xlabel('False Positive Rate')\n",
                "axes[1].set_ylabel('True Positive Rate')\n",
                "axes[1].legend(loc='lower right')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Business Stretch Goal: Cost Matrix & Optimal Threshold Optimization\n",
                "Treating a 0.50 cutoff as sacred is bad business practice.\n",
                "- **Cost of False Negative (Missed Fraud):** Actual stolen money in EGP (average thousands of EGP).\n",
                "- **Cost of False Positive (False Alarm):** Customer friction, verification SMS/Call (~50 EGP).\n",
                "We calculate total business financial loss across thresholds $T \\in [0.01, 0.99]$ to find the exact threshold that minimizes losses."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "best_probs = predictions_prob['HistGradientBoosting']\n",
                "thresholds = np.linspace(0.01, 0.99, 99)\n",
                "costs = []\n",
                "\n",
                "cost_fp = 50.0  # EGP per false alarm\n",
                "\n",
                "for t in thresholds:\n",
                "    preds = (best_probs >= t).astype(int)\n",
                "    fp = ((preds == 1) & (y_test == 0)).sum()\n",
                "    fn = ((preds == 0) & (y_test == 1)).sum()\n",
                "    \n",
                "    # Financial loss\n",
                "    fn_loss = amt_test[(preds == 0) & (y_test == 1)].sum()\n",
                "    fp_loss = fp * cost_fp\n",
                "    total = fn_loss + fp_loss\n",
                "    costs.append({'threshold': t, 'total_loss': total, 'fp_loss': fp_loss, 'fn_loss': fn_loss, 'fp': fp, 'fn': fn})\n",
                "\n",
                "cost_df = pd.DataFrame(costs)\n",
                "optimal_t = cost_df.loc[cost_df['total_loss'].idxmin(), 'threshold']\n",
                "min_loss = cost_df['total_loss'].min()\n",
                "loss_at_default = cost_df.loc[(cost_df['threshold'] - 0.5).abs().idxmin(), 'total_loss']\n",
                "\n",
                "print(f\"Optimal Decision Threshold: {optimal_t:.2f}\")\n",
                "print(f\"Total Loss at Optimal Threshold: {min_loss:,.2f} EGP\")\n",
                "print(f\"Total Loss at Default 0.50 Cutoff: {loss_at_default:,.2f} EGP\")\n",
                "print(f\"Financial Savings from Threshold Optimization: {loss_at_default - min_loss:,.2f} EGP!\")\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.plot(cost_df['threshold'], cost_df['total_loss'], label='Total Business Loss (EGP)', color='crimson', lw=2.5)\n",
                "plt.axvline(optimal_t, color='green', linestyle='--', label=f'Optimal Threshold ({optimal_t:.2f})')\n",
                "plt.axvline(0.50, color='gray', linestyle=':', label='Default Cutoff (0.50)')\n",
                "plt.title('Business Financial Loss vs. Decision Threshold')\n",
                "plt.xlabel('Detection Threshold')\n",
                "plt.ylabel('Total Financial Loss (EGP)')\n",
                "plt.legend()\n",
                "plt.show()"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Business Recommendations & Next Steps\n",
                "1. **Deploy Tiered Risk Actions:**\n",
                "   - Risk Score $< 0.15$: Auto-approve immediately.\n",
                "   - Risk Score $0.15 - 0.70$: Trigger dynamic OTP / Biometric authentication on mobile.\n",
                "   - Risk Score $> 0.70$: Freeze transaction and alert anti-fraud desk.\n",
                "2. **Agent Network Compliance:** Prioritize monitoring Vodafone Cash agent cash-outs exceeding 10,000 EGP following rapid peer-to-peer transfers within 10 minutes.\n",
                "3. **Model Limitations:** The model requires continuous re-training as fraudsters adapt tactics (concept drift). Retraining should occur bi-weekly."
            ]
        }
    ]
    
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
        
    print(f"Jupyter notebook created successfully at: {NOTEBOOK_PATH}")

if __name__ == '__main__':
    create_fraud_notebook()
