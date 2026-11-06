"""
End-to-End Orchestrator Pipeline
Mobile-Wallet Fraud Detection Engine (Vodafone Cash & InstaPay)
"""
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.data_preprocessing import prepare_train_test_dataset
from src.fraud_detection import train_and_evaluate_all

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def run_full_pipeline():
    print("=" * 70)
    print("🚀 FINTECH FRAUD DETECTION PIPELINE: VODAFONE CASH & INSTAPAY")
    print("=" * 70)
    
    # 1. Feature Engineering & Preprocessing
    print("\n[STEP 1/3] Preprocessing raw transactions & engineering balance features...")
    df_feat = prepare_train_test_dataset()
    
    # 2. Model Training & Cost-Loss Optimization
    print("\n[STEP 2/3] Training benchmarks & calculating cost-loss threshold curves...")
    metrics = train_and_evaluate_all()
    
    # 3. Generating Reproducible Jupyter Notebook
    print("\n[STEP 3/3] Generating clean Jupyter notebook for grading & portfolio...")
    from src.generate_notebook import create_fraud_notebook
    create_fraud_notebook()
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS: Mobile-Wallet Fraud Detection Pipeline Finished Successfully!")
    print(f"Artifacts saved in: {MODELS_DIR} and {DATA_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    run_full_pipeline()
