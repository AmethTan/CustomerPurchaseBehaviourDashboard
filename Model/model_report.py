import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap 
import json
import re
import random
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score, average_precision_score

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'Data', 'data.csv')
MODEL_PATH = BASE_DIR
REPORT_DIR = os.path.join(BASE_DIR, '..', 'Reports')

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

# Set plotting style
sns.set(style="whitegrid")

CONFIG = {
    'AVG_CUSTOMER_LIFESPAN_YEARS': 1.0 
}

# ==========================================
# 2. MODEL DEFINITIONS (SASRec)
# ==========================================
class SASRecModel(nn.Module):
    def __init__(self, num_items, maxlen, hidden_units, dropout_rate):
        super(SASRecModel, self).__init__()
        self.item_emb = nn.Embedding(num_items, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units, nhead=2, dim_feedforward=hidden_units*2, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear(hidden_units, num_items)
        self.maxlen = maxlen
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.item_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        src_key_padding_mask = (x == 0).all(dim=-1)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.out(x)

# ==========================================
# 3. ROBUST FEATURE ENGINEERING (Must match Retrain Logic)
# ==========================================
def engineer_features(df):
    print("   Engineering Features (Robust Logic)...")
    
    # Sort just like training
    df = df.sort_values(by=['user_id', 'event_time']).reset_index(drop=True)

    # Flags
    df['is_purchase'] = (df['event_type'] == 'purchase').astype(int)
    df['is_cart'] = (df['event_type'] == 'cart').astype(int)
    
    # --- LOGIC: CART ABANDONMENT (Strict) ---
    df['was_carted_in_session'] = df.groupby(['user_session', 'product_id'])['is_cart'].transform('max')
    df['is_cart_conversion'] = df['is_purchase'] * df['was_carted_in_session']
    
    # --- 1. Total Purchases ---
    df['total_purchases'] = df.groupby('user_id')['is_purchase'].cumsum().shift(1).fillna(0)
    
    # --- 2. Frequency Per Month ---
    user_start_date = df.groupby('user_id')['event_time'].transform('min')
    days_since_start = (df['event_time'] - user_start_date) / np.timedelta64(1, 'D')
    df['tenure_months'] = (days_since_start / 30.44) + 0.1
    df['frequency_per_month'] = df['total_purchases'] / df['tenure_months']

    # --- 3. Monetary Value (AOV) ---
    df['spend'] = df['price'] * df['is_purchase']
    total_spend = df.groupby('user_id')['spend'].cumsum().shift(1).fillna(0)
    df['aov'] = total_spend / df['total_purchases']
    df['aov'] = df['aov'].fillna(0)
    
    # --- 4. CLV (Formula) ---
    df['clv_formula'] = df['aov'] * df['frequency_per_month'] * 12 * CONFIG['AVG_CUSTOMER_LIFESPAN_YEARS']

    # --- 5. Cart Abandonment Rate (Strict) ---
    total_carts = df.groupby('user_id')['is_cart'].cumsum().shift(1).fillna(0)
    total_cart_conversions = df.groupby('user_id')['is_cart_conversion'].cumsum().shift(1).fillna(0)
    
    purchase_cart_ratio = total_cart_conversions / total_carts
    purchase_cart_ratio = purchase_cart_ratio.fillna(0) 
    
    df['cart_abandon_rate'] = 1.0 - purchase_cart_ratio
    df.loc[total_carts == 0, 'cart_abandon_rate'] = 0 
    df['cart_abandon_rate'] = df['cart_abandon_rate'].clip(0, 1)

    # --- 6. Recency (ROBUST GLOBAL SORT FIX) ---
    # A. Prepare "Purchases" dataframe
    purchases = df[df['is_purchase'] == 1][['user_id', 'event_time']].copy()
    purchases['last_buy'] = purchases['event_time'] 
    
    # B. Create TEMPORARY copies sorted by Time (Global) for merge_asof
    df_temp = df[['user_id', 'event_time']].sort_values('event_time')
    purchases_temp = purchases.sort_values('event_time')
    
    # C. Perform Merge
    df_merged = pd.merge_asof(
        df_temp, 
        purchases_temp, 
        on='event_time', 
        by='user_id', 
        direction='backward', 
        allow_exact_matches=False
    )
    
    # D. Sort back by Index
    df_merged = df_merged.sort_index()
    
    # E. Assign result
    df['last_buy_time'] = df_merged['last_buy']
    df['recency_days'] = (df['event_time'] - df['last_buy_time']).dt.total_seconds() / 86400
    df['recency_days'] = df['recency_days'].fillna(-1)
    
    # --- 7. Session Features ---
    df['next_event'] = df.groupby('user_session')['event_time'].shift(-1)
    df['time_on_page'] = (df['next_event'] - df['event_time']).dt.total_seconds().fillna(0)
    
    df['pages_viewed_session'] = df.groupby('user_session').cumcount() + 1
    
    # --- 8. Avg Pages Per Session ---
    df['new_sess_flag'] = (df['user_session'] != df.groupby('user_id')['user_session'].shift(1)).astype(int)
    total_sessions = df.groupby('user_id')['new_sess_flag'].cumsum()
    total_views = df.groupby('user_id').cumcount() + 1
    df['avg_pages_per_session'] = total_views / total_sessions

    # --- 9. Product Popularity ---
    df['product_page_views'] = df.groupby('product_id').cumcount() + 1

    return df

# ==========================================
# 4. REPORT GENERATION FUNCTIONS
# ==========================================

def generate_xgboost_report(df):
    print(">>> Generating XGBoost Report (Retrained Model)...")
    
    # 1. Feature Engineering
    df = engineer_features(df)
    
    # 2. Load Retrained Artifacts
    try:
        with open(os.path.join(MODEL_PATH, 'le_brand_retrained.pkl'), 'rb') as f: le_brand = pickle.load(f)
        with open(os.path.join(MODEL_PATH, 'le_cat_retrained.pkl'), 'rb') as f: le_cat = pickle.load(f)
        with open(os.path.join(MODEL_PATH, 'feature_names.pkl'), 'rb') as f: features = pickle.load(f)
        with open(os.path.join(MODEL_PATH, 'xgb_threshold.pkl'), 'rb') as f: threshold = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}")
        print("Did you run xgboost_retrain.py first?")
        return

    # Encoding
    def safe_transform(encoder, series):
        series = series.astype(str)
        return series.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    df['brand_encoded'] = safe_transform(le_brand, df['brand'])
    df['cat_encoded'] = safe_transform(le_cat, df['category_code'])
    
    # Prepare Data
    X = df[features].fillna(0)
    y = df['is_purchase']
    
    # Load Model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_PATH, 'xgboost_retrained.json'))
    
    # Predict
    print("   Calculating predictions...")
    y_probs = xgb_model.predict_proba(X)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)
    
    # --- METRICS CALCULATION ---
    acc = accuracy_score(y, y_preds)
    roc_auc = roc_curve(y, y_probs) # Just to get coords, score calculated below
    auc_score = auc(roc_auc[0], roc_auc[1])
    pr_auc = average_precision_score(y, y_probs)
    
    # --- PLOTS ---
    
    # A. Native Importance
    print("   Creating Importance Plots...")
    booster = xgb_model.get_booster()
    importance_types = ['gain', 'weight', 'cover']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Map f0, f1... to real names
    feature_map = {f'f{i}': name for i, name in enumerate(features)}

    for i, imp_type in enumerate(importance_types):
        scores = booster.get_score(importance_type=imp_type)
        mapped_scores = {feature_map.get(k, k): v for k, v in scores.items()}
        df_imp = pd.DataFrame(list(mapped_scores.items()), columns=['Feature', 'Score']).sort_values(by='Score', ascending=False)
        sns.barplot(x='Score', y='Feature', data=df_imp, ax=axes[i], palette='viridis', hue='Feature', legend=False)
        axes[i].set_title(f'Feature Importance: {imp_type.capitalize()}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'xgboost_variable_importance_native.png'))
    plt.close()
    
    # B. SHAP (JSON Fix applied)
    print("   Calculating SHAP Values...")
    if len(X) > 2000:
        X_sample = X.sample(2000, random_state=42)
    else:
        X_sample = X

    temp_json_path = os.path.join(MODEL_PATH, 'temp_shap_model.json')
    booster.save_model(temp_json_path)

    with open(temp_json_path, 'r') as f:
        model_json = f.read()
    
    # Fix the base_score format for SHAP
    model_json_fixed = re.sub(r'"base_score":"\[(.*?)\]"', r'"base_score":"\1"', model_json)

    with open(temp_json_path, 'w') as f:
        f.write(model_json_fixed)

    booster_fixed = xgb.Booster()
    booster_fixed.load_model(temp_json_path)

    try:
        explainer = shap.TreeExplainer(booster_fixed)
        shap_values = explainer.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'xgboost_shap_summary.png'))
        plt.close()
    except Exception as e:
        print(f"   Warning: SHAP failed: {e}")
    finally:
        if os.path.exists(temp_json_path): os.remove(temp_json_path)

    # C. Performance Curves
    print("   Creating Performance Curves...")
    fpr, tpr, _ = roc_curve(y, y_probs)
    precision, recall, _ = precision_recall_curve(y, y_probs)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.2f}')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[1].plot(recall, precision, color='green', lw=2, label=f'PR-AUC = {pr_auc:.2f}')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'xgboost_performance_curves.png'))
    plt.close()
    
    # --- TEXT REPORT ---
    report_txt = f"""
    XGBoost Final Model Report
    ==========================
    Optimal Threshold: {threshold:.4f}
    
    KEY PERFORMANCE METRICS
    -----------------------
    Accuracy:      {acc:.4f}  (Overall Correctness)
    ROC-AUC:       {auc_score:.4f}  (Ability to rank buyers vs non-buyers)
    PR-AUC:        {pr_auc:.4f}   (Precision-Recall Area - Critical for Imbalanced Data)
    
    Classification Report:
    {classification_report(y, y_preds)}
    
    Confusion Matrix:
    {confusion_matrix(y, y_preds)}
    
    Variable Definitions:
    ---------------------
    1. Recency: Days since previous purchase (Calculated dynamically).
    2. CLV Formula: AOV * Monthly Frequency * 12 Months.
    3. Cart Abandonment: 1 - (Purchases from Cart / Total Carts).
    """
    with open(os.path.join(REPORT_DIR, 'xgboost_report.txt'), 'w') as f:
        f.write(report_txt)
    print("   XGBoost Report Saved.")

def generate_fpgrowth_report():
    print(">>> Generating FP-Growth Report...")
    try:
        rules = pd.read_pickle(os.path.join(MODEL_PATH, 'fpgrowth_rules.pkl'))
        if rules.empty:
            print("   No rules to report.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="support", y="confidence", size="lift", hue="lift", data=rules, sizes=(20, 200), palette="coolwarm")
        plt.title("Association Rules: Support vs Confidence")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'fpgrowth_rules_scatter.png'))
        plt.close()
        
        with open(os.path.join(REPORT_DIR, 'fpgrowth_top_rules.txt'), 'w') as f:
            f.write(rules.head(10).to_string())
        print("   FP-Growth Report Saved.")
    except FileNotFoundError:
        print("   FP-Growth rules file not found.")

def generate_sasrec_report(df, k=10):
    print(f">>> Generating SASRec Performance Report (@K={k})...")
    
    # 1. Load Artifacts
    try:
        with open(os.path.join(MODEL_PATH, 'item_map.pkl'), 'rb') as f: item_map = pickle.load(f)
        num_items = len(item_map) + 1
        
        # Initialize Model Structure
        model = SASRecModel(num_items=num_items, maxlen=49, hidden_units=64, dropout_rate=0.1)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'sasrec_model.pth'), map_location='cpu'))
        model.eval()
    except Exception as e:
        print(f"   Could not load SASRec model: {e}")
        return

    # 2. Prepare User Sequences
    interest_events = df[df['event_type'].isin(['view', 'cart', 'purchase'])]
    user_groups = interest_events.groupby('user_id')['product_id'].apply(list)
    
    # Filter users with enough history (at least 2 items: 1 input, 1 target)
    valid_users = [u for u in user_groups if len(u) >= 2]
    
    # Sampling for performance
    sample_size = 500
    if len(valid_users) > sample_size:
        print(f"   Sampling {sample_size} users from {len(valid_users)} total valid users...")
        valid_users = random.sample(valid_users, sample_size)
    else:
        print(f"   Evaluating on all {len(valid_users)} valid users...")

    # 3. Metric Accumulators
    metrics = {
        'recall': [],    # Hit Rate
        'ndcg': [],      # Normalized Discounted Cumulative Gain
        'mrr': [],       # Mean Reciprocal Rank
        'precision': []  # Precision
    }

    print("   Running predictions...")
    with torch.no_grad():
        for history in valid_users:
            # Map raw product IDs to model integers
            seq_ids = [item_map.get(x, 0) for x in history]
            
            # Target is the LAST item; Input is everything before it
            target_item = seq_ids[-1]
            input_seq = seq_ids[:-1]
            
            # Pad sequence to match training length (maxlen=49)
            maxlen = 49
            input_seq = ([0] * (maxlen - len(input_seq))) + input_seq[-maxlen:]
            
            # Predict
            logits = model(torch.LongTensor([input_seq]))
            
            # Get Top-K Recommendations
            _, top_k_indices = torch.topk(logits[:, -1, :], k)
            top_k_list = top_k_indices[0].tolist()
            
            # --- CALCULATE METRICS PER USER ---
            
            if target_item in top_k_list:
                # 1. Rank (1-based index)
                rank = top_k_list.index(target_item) + 1
                
                # 2. Recall@K (Hit Rate) - 1 if present, 0 if not
                metrics['recall'].append(1.0)
                
                # 3. MRR - 1/Rank
                metrics['mrr'].append(1.0 / rank)
                
                # 4. NDCG@K - 1 / log2(rank + 1)
                metrics['ndcg'].append(1.0 / np.log2(rank + 1))
                
                # 5. Precision@K (1 correct item / K)
                metrics['precision'].append(1.0 / k)
                
            else:
                # Target not found in Top K
                metrics['recall'].append(0.0)
                metrics['mrr'].append(0.0)
                metrics['ndcg'].append(0.0)
                metrics['precision'].append(0.0)

    # 4. Aggregate Results
    avg_recall = np.mean(metrics['recall'])
    avg_ndcg = np.mean(metrics['ndcg'])
    avg_mrr = np.mean(metrics['mrr'])
    avg_precision = np.mean(metrics['precision'])
    
    # 5. Calculate F1-Score@K (Harmonic Mean of Avg Precision and Avg Recall)
    if (avg_precision + avg_recall) > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0.0

    # 6. Generate Report Text
    report_text = f"""
    SASRec Sequential Recommendation Report
    =======================================
    Config: Top-K = {k}
    Evaluated on {len(valid_users)} sampled user sequences.
    
    Metrics:
    --------
    1. Recall@{k} (Hit Rate):  {avg_recall:.4f}
       (Probability the correct next item is in the top {k} suggestions)
       
    2. NDCG@{k}:               {avg_ndcg:.4f}
       (Ranking quality measure; higher if correct item is higher in the list)
       
    3. MRR (Mean Recip. Rank): {avg_mrr:.4f}
       (Average of 1/rank; rewards appearing at the very top)
       
    4. Precision@{k}:          {avg_precision:.4f}
       (Percentage of the top {k} list that was relevant. 
        Note: Max possible is {1/k:.2f} since there is only 1 target item)
       
    5. F1-Score@{k}:           {f1_score:.4f}
       (Harmonic mean of Precision and Recall)
    """
    
    # Save to file
    report_path = os.path.join(REPORT_DIR, 'sasrec_performance_advanced.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
        
    print(f"   SASRec Report Saved to: {report_path}")
    print(f"   Summary: Recall@{k}: {avg_recall:.2%}, NDCG@{k}: {avg_ndcg:.4f}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print("Error: Data file not found.")
    else:
        df = pd.read_csv(DATA_PATH)
        df['event_time'] = pd.to_datetime(df['event_time']) # Pre-convert for speed
        
        generate_xgboost_report(df)
        generate_fpgrowth_report()
        generate_sasrec_report(df)
        
        print(f"\n✅ All reports generated in: {os.path.abspath(REPORT_DIR)}")