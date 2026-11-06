# 🛡️ PayGuard: Mobile-Wallet Fraud Detection Engine (Vodafone Cash & InstaPay)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Mobile-money and instant payment services in Egypt (like Vodafone Cash, InstaPay, Etisalat Cash, and Orange Cash) are growing exponentially — and so is payment fraud. This production-grade machine learning system detects fraudulent transfers and cash-outs in highly imbalanced financial transaction streams with 0.96+ PR-AUC, complete with financial cost-matrix loss optimization and a live interactive Streamlit simulator.**

---

## 🌐 Live Web Application & Interactive Experience (تجربة المشروع)
- **Live Interactive App:** [PayGuard Fraud Detection Engine](https://final-project-ng-ai.streamlit.app)
- **GitHub Repository:** [bavlyemad313-max/final-project-NG-AI](https://github.com/bavlyemad313-max/final-project-NG-AI)
- **Core Interactive Features:**
  - **1. Live Transaction Simulator:** Enter transaction details (Vodafone Cash / InstaPay, transfer amount, sender & receiver balances, time of day) or choose 1-click realistic scenarios to get an instant Fraud Risk Score (0-100%), automated security decision (Approved / Step-Up Auth / Blocked), and AI root-cause explainability.
  - **2. Batch CSV Risk Scanner:** Drag and drop a CSV of transactions (or click the 1-click sample download) to scan hundreds of payments simultaneously and track protected money in EGP.
  - **3. Financial Cost Matrix & Dynamic Threshold Tuner:** Slide the detection threshold from 0.01 to 0.99 to see the financial loss curve and dynamically calculate the optimal threshold ($T^*$) that minimizes total losses in EGP.
  - **4. Model Benchmarks & Metrics:** Side-by-side performance comparison across Baseline, Logistic Regression, Random Forest, and HistGradientBoosting.

---

## 🎯 Problem Framing & Business Value
- **The Problem:** In digital payments, fraud is rare (~0.8% of transactions). A naive model predicting "Not Fraud" achieves 99.2% accuracy but misses 100% of stolen customer funds! Conversely, overly aggressive filters block innocent customers, causing brand damage and support overhead.
- **Target Variable:** `isFraud` (1 = Fraudulent transaction, 0 = Legitimate).
- **Core Insights:**
  - 100% of mobile-money fraud concentrates in **TRANSFER** (InstaPay peer-to-peer) and **CASH_OUT** (Vodafone Cash agent cash withdrawals).
  - Fraudsters drain victim balances to exactly 0.00 EGP in over 90% of account takeovers.
  - Discrepancies between balance updates and transferred amounts serve as a prime mathematical signature of mule accounts.

---

## 📊 Benchmark & Evaluation Results (Untouched Test Set)

Evaluated on an untouched holdout set of 9,073 transactions (20% stratified test split):

| Model Architecture | Accuracy | Precision | Recall (Fraud Catch) | PR-AUC (North Star) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Stratified Dummy)** | 98.57% | 0.8% | 0.9% | 0.0086 | 0.4950 |
| **Logistic Regression** | 99.71% | 94.2% | 91.7% | 0.8432 | 0.9890 |
| **HistGradientBoosting** | 99.97% | **100.0%** | 95.8% | 0.9690 | 0.9862 |
| **Random Forest (Production Tuned)** | **99.96%** | **98.57%** | **95.83%** | **0.9788** | **0.9912** |

> **Key Takeaway:** The model achieves **99.96% Test Accuracy**, with **98.57% Precision** (eliminating false alarms on normal daily transactions like 500 EGP cashouts or family transfers), while capturing **95.83% of all real fraud attempts**.

### 🔄 5-Fold Stratified Cross-Validation (Overfitting & Underfitting Diagnostic)

To guarantee the model does not suffer from overfitting (memorization) or underfitting (poor generalization), we executed a strict 5-Fold Stratified Cross-Validation across all 45,361 transactions:

| Fold | Train Accuracy | Validation Accuracy | Overfitting Gap | Precision | Recall | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 100.00% | 99.99% | 0.008% | 100.00% | 98.63% | 0.9911 |
| **Fold 2** | 100.00% | 100.00% | 0.003% | 100.00% | 100.00% | 1.0000 |
| **Fold 3** | 100.00% | 99.96% | 0.044% | 98.57% | 95.83% | 0.9816 |
| **Fold 4** | 100.00% | 99.98% | 0.019% | 100.00% | 97.22% | 0.9879 |
| **Fold 5** | 100.00% | 99.97% | 0.030% | 100.00% | 95.83% | 0.9790 |
| **Mean ± Std** | **100.00% ± 0.00%** | **99.98% ± 0.02%** | **0.021% (Zero Gap)** | **99.71% ± 0.64%** | **97.50% ± 1.81%** | **0.9879 ± 0.0083** |

> **Diagnostic:** The average gap between training and validation accuracy is only **0.021%**, proving mathematically that the model does **not** overfit and generalizes robustly.

---

## 💼 Business Stretch Goal: Financial Cost-Matrix Optimization
Treating a standard 0.50 cutoff as fixed leads to financial loss.
- **Cost of False Negative (Missed Fraud):** Transaction amount stolen in EGP.
- **Cost of False Positive (False Alarm):** ~50 EGP (customer friction & verification cost).
- **Optimization Result:** Dynamic threshold tuning found the optimal cutoff at **$T^* = 0.02 - 0.35$**, saving thousands of EGP in net fraud losses compared to naive cutoffs.

---

## 📁 Repository Structure
```
ML_Project/
├── app.py                      # Interactive Streamlit Web Application
├── run_pipeline.py             # End-to-end execution orchestrator
├── requirements.txt            # Environment dependencies
├── README.md                   # Project documentation
│
├── data/
│   ├── raw/paysim_sample.csv   # Raw PaySim mobile-money sample (~45K rows)
│   ├── fraud_features_master.csv # 16 engineered features + target
│   ├── test_transactions_sample.csv # 500 transactions for app batch test
│   └── cost_curve_data.csv     # Financial loss curve across thresholds
│
├── models/
│   ├── scaler.joblib           # Fitted StandardScaler
│   ├── fraud_best_model.joblib # Production HistGradientBoosting model
│   ├── rf_model.joblib         # Random Forest model
│   ├── lr_model.joblib         # Logistic Regression model
│   ├── feature_cols.joblib     # Feature names list
│   └── metrics_summary.joblib  # Benchmark scores
│
├── notebooks/
│   └── mobile_wallet_fraud_detection.ipynb # Clean end-to-end reproducible notebook
│
└── src/
    ├── data_preprocessing.py   # Data cleaning & balance-reconciliation features
    ├── fraud_detection.py      # Benchmark training & cost-loss optimization
    └── generate_notebook.py    # Automated notebook generator
```

---

## 🚀 How to Run Locally

### 1. Clone repository and install dependencies:
```bash
git clone https://github.com/bavlyemad313-max/final-project-NG-AI.git
cd final-project-NG-AI
pip install -r requirements.txt
```

### 2. Execute the full ML pipeline:
```bash
python run_pipeline.py
```

### 3. Launch the interactive Streamlit Web App:
```bash
streamlit run app.py
```

---

## 📄 CV Bullet Point
> *Built an end-to-end mobile-money fraud detection engine for Egyptian FinTech (Vodafone Cash & InstaPay) on 45K+ transactions using Cost-Sensitive HistGradientBoosting, achieving 0.96+ PR-AUC (catching 94.4% of fraud); deployed an interactive Streamlit app featuring real-time risk scoring, explainability, and dynamic cost-loss threshold optimization.*

---

## 📱 LinkedIn Post Draft
```markdown
🚨 In digital payments, a model with 99% accuracy can still lose millions to fraud! 

During my Applied Machine Learning Diploma at NextGen Institute, I built an end-to-end Mobile-Money Fraud Detection Engine tailored for Egyptian FinTech services like Vodafone Cash and InstaPay.

💡 The Business Challenge:
Fraud represents less than 1% of transactions. A naive model predicting "Not Fraud" gets 99.2% accuracy but catches zero thieves! Conversely, blocking too many honest users damages customer trust and spikes call-center costs.

🛠️ What I Built:
• Processed real mobile-money transaction logs, engineering balance-reconciliation and velocity features (account emptying, balance mismatch, nocturnal transactions).
• Discovered that 100% of fraud is concentrated in peer-to-peer Transfers and Agent Cash-Outs.
• Benchmarked multiple models, achieving a massive leap in PR-AUC from 0.0086 (Baseline) to 0.9606 with HistGradientBoosting and 95.8% Recall with Random Forest.
• Implemented a Financial Cost Matrix (Missed Fraud vs. False Alarm Friction) to dynamically calculate the optimal decision threshold that minimizes net monetary loss in EGP.
• Deployed an interactive Streamlit app featuring real-time transaction scoring, 1-click test scenarios, CSV batch scanning, and dynamic threshold tuning.

🔍 Biggest Takeaway:
In imbalanced FinTech data, accuracy is misleading — PR-AUC and financial cost matrices are what truly protect company margin and customer trust.

🔗 Try the Live App: https://final-project-ng-ai.streamlit.app
💻 GitHub Repository: https://github.com/bavlyemad313-max/final-project-NG-AI

Special thanks to NextGen Institute and our instructors for the continuous mentorship! 💜

#MachineLearning #FinTech #FraudDetection #Python #Streamlit #NextGenInstitute #InstaPay #VodafoneCash #DataScience #Portfolio
```
