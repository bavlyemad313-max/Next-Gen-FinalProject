# 🛡️ PayGuard: Mobile-Wallet Fraud Detection Engine (Vodafone Cash & InstaPay)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Mobile-money and instant payment services in Egypt (like Vodafone Cash, InstaPay, Etisalat Cash, and Orange Cash) are growing exponentially — and so is payment fraud. This production-grade machine learning system detects fraudulent transfers and cash-outs in highly imbalanced financial transaction streams with 0.96+ PR-AUC, complete with financial cost-matrix loss optimization and a live interactive Streamlit simulator.**

---

## 🌐 Live Web Application & Interactive Experience (تجربة المشروع)
- **Live Interactive App:** [PayGuard Fraud Detection Engine](https://final-project-ng-ai-hyxfn77pd6m9ong84rkyy2.streamlit.app/)
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

| Model Architecture | PR-AUC (North Star) | ROC-AUC | Recall (Fraud Catch) | Precision | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Stratified Dummy)** | 0.0086 | 0.4950 | 0.9% | 0.8% | 0.0085 |
| **Logistic Regression (Class-Balanced)** | 0.8165 | **0.9907** | 93.1% | 7.2% | 0.1336 |
| **Random Forest (100 Trees, Balanced)** | 0.9587 | 0.9820 | **95.8%** | 88.5% | **0.9200** |
| **HistGradientBoosting (Production Tuned)** | **0.9606** | 0.9862 | 94.4% | **89.5%** | 0.9189 |

> **Key Takeaway:** In severe class imbalance, standard accuracy is misleading. **PR-AUC jumped from 0.0086 to 0.9606**, catching >94% of fraud while maintaining high precision to minimize customer friction.

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

🔗 Try the Live App: https://final-project-ng-ai-hyxfn77pd6m9ong84rkyy2.streamlit.app/
💻 GitHub Repository: https://github.com/bavlyemad313-max/final-project-NG-AI

Special thanks to NextGen Institute and our instructors for the continuous mentorship! 💜

#MachineLearning #FinTech #FraudDetection #Python #Streamlit #NextGenInstitute #InstaPay #VodafoneCash #DataScience #Portfolio
```
