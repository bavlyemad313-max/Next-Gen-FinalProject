"""
Mobile-Wallet Fraud Detection Engine (Vodafone Cash & InstaPay)
Data Preprocessing & Feature Engineering Pipeline
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'paysim_sample.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'fraud_features_master.csv')

def load_and_clean_data(file_path=RAW_DATA_PATH):
    """
    Loads raw PaySim mobile money transaction data and applies initial validation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at {file_path}")
    
    print(f"Loading raw mobile-money transactions from: {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Raw transactions loaded: {len(df):,} rows")
    
    df = df.dropna(subset=['amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']).copy()
    df['isFraud'] = df['isFraud'].astype(int)
    
    return df

def engineer_features(df):
    """
    Engineers robust, domain-specific financial features tailored for mobile wallets (Vodafone Cash / InstaPay):
    1. error_orig: Discrepancy in origin balance
    2. error_dest: Discrepancy in destination balance
    3. amount_to_orig_ratio: Ratio of transaction amount to sender balance
    4. is_account_emptied: Flag if sender balance was reduced to zero
    5. is_large_drain: Specific fraud signature of completely emptying a large balance (>= 10,000 EGP)
    6. dest_zero_after_tx: Suspicious signature where money is transferred but recipient balance remains zero (mule account)
    7. hour_of_day: Extracted hour (step % 24)
    8. is_night_transaction: High-risk nocturnal hours (11 PM - 5 AM)
    9. is_transfer, is_cash_out, is_payment, is_cash_in: One-hot encoded types
    10. dest_is_merchant: Flag if recipient is official merchant
    """
    print("Engineering robust FinTech fraud detection features...")
    data = df.copy()
    
    # 1. Balance Discrepancies
    data['error_orig'] = data['newbalanceOrig'] + data['amount'] - data['oldbalanceOrg']
    data['error_dest'] = data['oldbalanceDest'] + data['amount'] - data['newbalanceDest']
    
    # 2. Ratio of Transaction to Initial Balance
    data['amount_to_orig_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1.0)
    
    # 3. Account Emptied
    data['is_account_emptied'] = ((data['newbalanceOrig'] == 0) & (data['amount'] > 0)).astype(int)
    
    # 4. Large Account Drained (The genuine account takeover phishing signature in FinTech)
    data['is_large_drain'] = ((data['newbalanceOrig'] == 0) & (data['amount'] >= 10000)).astype(int)
    
    # 5. Mule Account Target (Money transferred into burner account that does not record balance)
    data['dest_zero_after_tx'] = ((data['newbalanceDest'] == 0) & (data['amount'] > 1000)).astype(int)
    
    # 6. Temporal Features
    data['hour_of_day'] = data['step'] % 24
    data['is_night_transaction'] = data['hour_of_day'].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
    
    # 7. Transaction Type Indicators
    data['is_transfer'] = (data['type'] == 'TRANSFER').astype(int)
    data['is_cash_out'] = (data['type'] == 'CASH_OUT').astype(int)
    data['is_payment'] = (data['type'] == 'PAYMENT').astype(int)
    data['is_cash_in'] = (data['type'] == 'CASH_IN').astype(int)
    
    # 8. Recipient Entity
    if 'nameDest' in data.columns:
        data['dest_is_merchant'] = data['nameDest'].astype(str).str.startswith('M').astype(int)
    else:
        data['dest_is_merchant'] = 0
        
    return data

def prepare_train_test_dataset():
    df = load_and_clean_data()
    df_feat = engineer_features(df)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df_feat.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")
    print(f"Total Transactions: {len(df_feat):,} | Fraudulent: {df_feat['isFraud'].sum():,} ({df_feat['isFraud'].mean()*100:.2f}%)")
    
    return df_feat

if __name__ == '__main__':
    prepare_train_test_dataset()
