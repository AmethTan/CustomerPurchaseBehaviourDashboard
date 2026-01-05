import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.combine import SMOTEENN

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'Data', 'data.csv')
MODEL_PATH = BASE_DIR

CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'use_gpu': False, 
    'AVG_CUSTOMER_LIFESPAN_YEARS': 1.0 
}

def load_data():
    print("--- Loading Data ---")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # 1. Sort by User -> Time for Cumulative features
    df = df.sort_values(by=['user_id', 'event_time']).reset_index(drop=True)
    
    df['brand'] = df['brand'].fillna('unknown')
    df['category_code'] = df['category_code'].fillna('unknown')
    
    print(f"Data Loaded. Shape: {df.shape}")
    return df

# ==========================================
# 2. ADVANCED FEATURE ENGINEERING (ROBUST)
# ==========================================
def engineer_features(df):
    print("--- Engineering Features (Robust Logic) ---")
    
    # Flags
    df['is_purchase'] = (df['event_type'] == 'purchase').astype(int)
    df['is_cart'] = (df['event_type'] == 'cart').astype(int)
    
    # --- LOGIC: CART ABANDONMENT (Strict) ---
    print("   Logic Check: Identifying Cart-to-Purchase flows...")
    df['was_carted_in_session'] = df.groupby(['user_session', 'product_id'])['is_cart'].transform('max')
    df['is_cart_conversion'] = df['is_purchase'] * df['was_carted_in_session']
    
    # --- 1. Total Purchases (Cumulative) ---
    print("   Calculating: Total Purchases...")
    df['total_purchases'] = df.groupby('user_id')['is_purchase'].cumsum().shift(1).fillna(0)
    
    # --- 2. Frequency Per Month (Fixed 'M' Error) ---
    print("   Calculating: Frequency/Month...")
    user_start_date = df.groupby('user_id')['event_time'].transform('min')
    
    # Convert time delta to Days ('D') then divide by avg days in month (30.44)
    days_since_start = (df['event_time'] - user_start_date) / np.timedelta64(1, 'D')
    df['tenure_months'] = (days_since_start / 30.44) + 0.1
    
    df['frequency_per_month'] = df['total_purchases'] / df['tenure_months']

    # --- 3. Monetary Value (AOV) ---
    print("   Calculating: AOV...")
    df['spend'] = df['price'] * df['is_purchase']
    total_spend = df.groupby('user_id')['spend'].cumsum().shift(1).fillna(0)
    df['aov'] = total_spend / df['total_purchases']
    df['aov'] = df['aov'].fillna(0)
    
    # --- 4. CLV (Formula) ---
    print("   Calculating: CLV Formula...")
    df['clv_formula'] = df['aov'] * df['frequency_per_month'] * 12 * CONFIG['AVG_CUSTOMER_LIFESPAN_YEARS']

    # --- 5. Cart Abandonment Rate (Strict) ---
    print("   Calculating: Cart Abandonment (Strict)...")
    total_carts = df.groupby('user_id')['is_cart'].cumsum().shift(1).fillna(0)
    total_cart_conversions = df.groupby('user_id')['is_cart_conversion'].cumsum().shift(1).fillna(0)
    
    purchase_cart_ratio = total_cart_conversions / total_carts
    purchase_cart_ratio = purchase_cart_ratio.fillna(0) 
    
    df['cart_abandon_rate'] = 1.0 - purchase_cart_ratio
    df.loc[total_carts == 0, 'cart_abandon_rate'] = 0 
    df['cart_abandon_rate'] = df['cart_abandon_rate'].clip(0, 1)

    # --- 6. Recency (ROBUST GLOBAL SORT FIX) ---
    print("   Calculating: Recency (Dataset Relative)...")
    
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
    
    # D. Sort back by Index to align with original 'df'
    df_merged = df_merged.sort_index()
    
    # E. Assign result
    df['last_buy_time'] = df_merged['last_buy']
    df['recency_days'] = (df['event_time'] - df['last_buy_time']).dt.total_seconds() / 86400
    df['recency_days'] = df['recency_days'].fillna(-1)
    
    df = df.drop(columns=['last_buy_time'])

    # --- 7. Session Features ---
    print("   Calculating: Session Features...")
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
# 3. TRAINING PROCESS (OPTIMIZED)
# ==========================================
def train_xgboost():
    # 1. Load Data
    df = load_data()
    df = engineer_features(df)
    
    print("\n--- Preparing Training Data ---")
    
    le_brand = LabelEncoder()
    df['brand_encoded'] = le_brand.fit_transform(df['brand'].astype(str))
    
    le_cat = LabelEncoder()
    df['cat_encoded'] = le_cat.fit_transform(df['category_code'].astype(str))
    
    features = [
        'total_purchases',       
        'frequency_per_month',   
        'aov',                  
        'clv_formula',          
        'cart_abandon_rate',     
        'recency_days',         
        'time_on_page',
        'pages_viewed_session',
        'avg_pages_per_session', 
        'product_page_views',    
        'price',
        'brand_encoded', 
        'cat_encoded'
    ]
    
    print(f"Features used: {features}")
    
    X = df[features].fillna(0)
    y = df['is_purchase']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y
    )
    
    # 2. Load Optimized Hyperparameters
    print("\n--- Loading Optimized Hyperparameters ---")
    params_path = os.path.join(MODEL_PATH, 'best_hyperparameters.pkl')
    flag_path = os.path.join(MODEL_PATH, 'use_smote_flag.pkl')
    
    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            best_params = pickle.load(f)
        print("   Loaded Best Params successfully.")
    else:
        print("   Warning: 'best_hyperparameters.pkl' not found. Using defaults.")
        ratio = float(np.sum(y == 0)) / np.sum(y == 1)
        best_params = {
            'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
            'scale_pos_weight': ratio, 'eval_metric': 'aucpr'
        }

    # 3. Check for SMOTE Requirement
    use_smote = False
    if os.path.exists(flag_path):
        with open(flag_path, 'rb') as f:
            use_smote = pickle.load(f)
            
    if use_smote:
        print("   Strategy: SMOTE-ENN (Applying resampling...)")
        smote = SMOTEENN(random_state=CONFIG['random_state'])
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   Resampled Train Shape: {X_train.shape}")
    else:
        print("   Strategy: Weight Only (No resampling needed)")

    # 4. Train
    print("\n--- Training XGBoost Model ---")
    model = xgb.XGBClassifier(**best_params)
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=False
    )
    
    # 5. Evaluate
    print("\n--- Evaluation (Test Set) ---")
    probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate PR-AUC (The metric we optimized for)
    pr_auc = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    
    print(f"PR-AUC (Primary Metric): {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Find Optimal Threshold for Classification Report
    # We pick threshold that maximizes F1 score
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_thresh_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_thresh_idx]
    
    print(f"Optimal Threshold (Max F1): {best_thresh:.4f}")
    preds = (probs >= best_thresh).astype(int)
    print(classification_report(y_test, preds))
    
    # 6. Save Artifacts
    print("\n--- Saving Artifacts ---")
    model.save_model(os.path.join(MODEL_PATH, 'xgboost_retrained.json'))
    
    # Save the optimal threshold! Important for the dashboard.
    with open(os.path.join(MODEL_PATH, 'xgb_threshold.pkl'), 'wb') as f: 
        pickle.dump(best_thresh, f)
        
    with open(os.path.join(MODEL_PATH, 'feature_names.pkl'), 'wb') as f: pickle.dump(features, f)
    with open(os.path.join(MODEL_PATH, 'le_brand_retrained.pkl'), 'wb') as f: pickle.dump(le_brand, f)
    with open(os.path.join(MODEL_PATH, 'le_cat_retrained.pkl'), 'wb') as f: pickle.dump(le_cat, f)

    print("Retraining Complete.")

if __name__ == "__main__":
    train_xgboost()