"""
Customer Segmentation Engine: RFM Clustering & PCA
Algorithms: K-Means, Hierarchical Clustering, PCA
Evaluation: Silhouette Score, Davies-Bouldin Index, Elbow Method
"""
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def log_transform_rfm(df: pd.DataFrame, features=['recency', 'frequency', 'monetary']):
    """
    Applies log1p transform to skewed RFM features.
    """
    df_transformed = df.copy()
    for feat in features:
        df_transformed[f'{feat}_log'] = np.log1p(np.maximum(0, df[feat]))
    return df_transformed

def evaluate_kmeans_k_range(X_scaled, k_range=range(2, 9)):
    """
    Computes Inertia and Silhouette scores across a range of k values
    to find the optimal number of clusters.
    """
    inertias = []
    silhouettes = []
    print("\n--- Evaluating K-Means Across K Range ---")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        inertias.append(km.inertia_)
        silhouettes.append(sil)
        print(f"k={k} | Inertia={km.inertia_:.1f} | Silhouette Score={sil:.4f}")
    return inertias, silhouettes

def label_segments_by_business_rules(rfm_summary: pd.DataFrame) -> dict:
    """
    Maps cluster IDs to intuitive business segment personas based on their
    mean Recency, Frequency, and Monetary percentiles:
    - Champions / VIP
    - Loyal Customers
    - Potential / New Shoppers
    - At-Risk Customers
    - Lost / Hibernating
    """
    # Sort clusters by monetary value to assign coherent personas
    sorted_clusters = rfm_summary.sort_values(by=['monetary', 'frequency'], ascending=False).index.tolist()
    
    mapping = {}
    if len(sorted_clusters) == 5:
        mapping[sorted_clusters[0]] = "Champions (VIP)"
        mapping[sorted_clusters[1]] = "Loyal Customers"
        mapping[sorted_clusters[2]] = "Potential Loyalists"
        mapping[sorted_clusters[3]] = "At-Risk Customers"
        mapping[sorted_clusters[4]] = "Lost / Hibernating"
    elif len(sorted_clusters) == 4:
        mapping[sorted_clusters[0]] = "Champions (VIP)"
        mapping[sorted_clusters[1]] = "Loyal Customers"
        mapping[sorted_clusters[2]] = "At-Risk Customers"
        mapping[sorted_clusters[3]] = "Lost / Hibernating"
    else:
        for idx, c in enumerate(sorted_clusters):
            mapping[c] = f"Segment {idx+1}"
            
    return mapping

def segment_action_strategy():
    """
    Returns marketing strategies and playbooks for each customer segment persona.
    """
    return {
        "Champions (VIP)": {
            "action": "Exclusive VIP rewards, early access to new lines, premium personalized support.",
            "discount_strategy": "Zero or minimal discounts; focus on prestige and VIP exclusivity.",
            "channel": "Dedicated VIP emails & WhatsApp concierge."
        },
        "Loyal Customers": {
            "action": "Loyalty program points, cross-sell related products, referral incentives.",
            "discount_strategy": "Moderate bundle discounts (e.g. buy 2 get 1 at 20% off).",
            "channel": "Personalized email recommendations."
        },
        "Potential Loyalists": {
            "action": "Welcome onboarding series, membership offers, reviews request.",
            "discount_strategy": "Time-sensitive second-purchase discount coupon (10-15%).",
            "channel": "Email and app push notifications."
        },
        "At-Risk Customers": {
            "action": "Aggressive win-back campaigns, customer feedback surveys, reactivation offers.",
            "discount_strategy": "High discount (20-30%) or free shipping gift coupon.",
            "channel": "SMS and targeted re-engagement emails."
        },
        "Lost / Hibernating": {
            "action": "Low-frequency seasonal newsletters, clearance sales, or remove from active ad spend.",
            "discount_strategy": "Heavy liquidation / clearance offers only.",
            "channel": "Automated monthly digest only."
        }
    }

def train_segmentation(df_customer: pd.DataFrame, n_clusters=4):
    os.makedirs(MODELS_DIR, exist_ok=True)
    rfm_cols = ['recency', 'frequency', 'monetary']
    
    # 1. Log Transform
    df_log = log_transform_rfm(df_customer, rfm_cols)
    log_cols = [f'{c}_log' for c in rfm_cols]
    
    # 2. Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log[log_cols])
    
    # 3. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # 4. Hierarchical Clustering comparison
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hierarchical.fit_predict(X_scaled)
    
    # 5. Metrics
    km_sil = silhouette_score(X_scaled, kmeans_labels)
    km_db = davies_bouldin_score(X_scaled, kmeans_labels)
    hier_sil = silhouette_score(X_scaled, hier_labels)
    hier_db = davies_bouldin_score(X_scaled, hier_labels)
    
    print("\n--- Clustering Evaluation Comparison ---")
    print(f"K-Means (k={n_clusters}): Silhouette={km_sil:.4f} | Davies-Bouldin={km_db:.4f}")
    print(f"Hierarchical (k={n_clusters}): Silhouette={hier_sil:.4f} | Davies-Bouldin={hier_db:.4f}")
    
    # 6. PCA (2 components for 2D visualization)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    df_customer = df_customer.copy()
    df_customer['cluster'] = kmeans_labels
    df_customer['pca_x'] = X_pca[:, 0]
    df_customer['pca_y'] = X_pca[:, 1]
    
    # Calculate Cluster Profiles
    summary = df_customer.groupby('cluster')[rfm_cols].mean()
    cluster_names = label_segments_by_business_rules(summary)
    df_customer['segment_name'] = df_customer['cluster'].map(cluster_names)
    
    # Save artifacts
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'rfm_scaler.joblib'))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans_model.joblib'))
    joblib.dump(pca, os.path.join(MODELS_DIR, 'pca_model.joblib'))
    joblib.dump(cluster_names, os.path.join(MODELS_DIR, 'cluster_names.joblib'))
    
    print("\n--- Final Customer Segments Profile ---")
    final_summary = df_customer.groupby('segment_name').agg(
        customer_count=('customer_id', 'count'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean'),
        total_revenue=('monetary', 'sum'),
        target_90d_revenue=('target_spend_90d', 'sum')
    ).reset_index()
    print(final_summary)
    
    return df_customer, final_summary

if __name__ == '__main__':
    customer_csv = os.path.join(DATA_DIR, 'customer_rfm_clv.csv')
    if os.path.exists(customer_csv):
        df_cust = pd.read_csv(customer_csv)
        train_segmentation(df_cust)
    else:
        print(f"Run data_preprocessing.py first; {customer_csv} not found.")
