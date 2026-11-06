"""
Mobile-Wallet Fraud Detection Engine (Vodafone Cash & InstaPay)
Data Preprocessing & Feature Engineering Pipeline
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    
    # Basic data integrity checks
    df = df.dropna(subset=['amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']).copy()
    df['isFraud'] = df['isFraud'].astype(int)
    
    return df

def engineer_features(df):
    """
    Engineers domain-specific financial features tailored for mobile wallets (Vodafone Cash / InstaPay):
    1. error_orig: Discrepancy between old balance, amount sent, and new balance
    2. error_dest: Discrepancy between recipient's old balance, amount received, and new balance
    3. amount_to_orig_ratio: Ratio of transaction amount to sender's initial balance
    4. is_account_emptied: Flag indicating if the sender wallet balance was reduced to zero
    5. hour_of_day: Extracted from transaction step (step % 24)
    6. day_of_week: Day cycle (step // 24 % 7)
    7. is_night_transaction: Transactions between 11 PM and 6 AM
    8. is_transfer: Flag for peer-to-peer / InstaPay transfers
    9. is_cash_out: Flag for agent cash-out / ATM withdrawals
    10. dest_is_merchant: Flag if the recipient is a merchant account
    """
    print("Engineering mobile-money fraud detection features...")
    data = df.copy()
    
    # 1. Origin and Destination Balance Inconsistencies (Key FinTech fraud indicator)
    data['error_orig'] = data['newbalanceOrig'] + data['amount'] - data['oldbalanceOrg']
    data['error_dest'] = data['oldbalanceDest'] + data['amount'] - data['newbalanceDest']
    
    # 2. Transaction Amount to Sender Balance Ratio
    data['amount_to_orig_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1.0)
    
    # 3. Account Completely Emptied Flag (Common in account takeover / phishing)
    data['is_account_emptied'] = ((data['newbalanceOrig'] == 0) & (data['amount'] > 0)).astype(int)
    
    # 4. Temporal Features
    data['hour_of_day'] = data['step'] % 24
    data['day_of_week'] = (data['step'] // 24) % 7
    data['is_night_transaction'] = data['hour_of_day'].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
    
    # 5. Transaction Type Flags (Domain Knowledge: in mobile wallets, frauds concentrate in Transfers & Cash-Outs)
    data['is_transfer'] = (data['type'] == 'TRANSFER').astype(int)
    data['is_cash_out'] = (data['type'] == 'CASH_OUT').astype(int)
    data['is_payment'] = (data['type'] == 'PAYMENT').astype(int)
    data['is_cash_in'] = (data['type'] == 'CASH_IN').astype(int)
    
    # 6. Recipient Type (Merchant vs Individual)
    if 'nameDest' in data.columns:
        data['dest_is_merchant'] = data['nameDest'].astype(str).str.startswith('M').astype(int)
    else:
        data['dest_is_merchant'] = 0
        
    return data

def prepare_train_test_dataset(test_size=0.2, random_state=42):
    """
    Full pipeline to clean, engineer features, and save processed dataset.
    """
    df = load_and_clean_data()
    df_feat = engineer_features(df)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df_feat.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")
    print(f"Total Transactions: {len(df_feat):,} | Fraudulent: {df_feat['isFraud'].sum():,} ({df_feat['isFraud'].mean()*100:.2f}%)")
    
    return df_feat

if __name__ == '__main__':
    prepare_train_test_dataset()
