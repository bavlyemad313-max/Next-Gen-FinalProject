"""
Dataset Download and Preparation Script for Online Retail II
"""
import os
import urllib.request
import zipfile
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_ZIP = os.path.join(DATA_DIR, 'online_retail_ii.zip')
EXTRACT_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_CSV = os.path.join(DATA_DIR, 'online_retail_clean.csv')

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

def download_data():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    if os.path.exists(PROCESSED_CSV):
        print(f"Processed dataset already exists at {PROCESSED_CSV}")
        return

    if not os.path.exists(RAW_ZIP):
        print(f"Downloading from UCI repository: {UCI_URL} ...")
        urllib.request.urlretrieve(UCI_URL, RAW_ZIP)
        print("Download complete.")

    print("Extracting ZIP archive...")
    with zipfile.ZipFile(RAW_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"Extracted to {EXTRACT_DIR}")

if __name__ == "__main__":
    download_data()
