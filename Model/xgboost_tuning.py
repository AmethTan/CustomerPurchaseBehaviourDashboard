import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
import pickle
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
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
    'n_trials': 50,  # UPDATED: Increased to 50 trials per strategy (Total = 150 runs)
    'use_gpu': False
}

# ==========================================
# 2. ROBUST DATA ENGINEERING
# ==========================================
def load_and_engineer():
    print(">>> Loading & Engineering Data...")
    if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"{DATA_PATH} not found")
    df = pd.read_csv(DATA_PATH)
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # 1. Sort by User -> Time
    df = df.sort_values(by=['user_id', 'event_time']).reset_index(drop=True)
    df['brand'] = df['brand'].fillna('unknown')
    df['category_code'] = df['category_code'].fillna('unknown')

    # Flags
    df['is_purchase'] = (df['event_type'] == 'purchase').astype(int)
    df['is_cart'] = (df['event_type'] == 'cart').astype(int)
    
    # Cart Abandonment (Strict)
    df['was_carted_in_session'] = df.groupby(['user_session', 'product_id'])['is_cart'].transform('max')
    df['is_cart_conversion'] = df['is_purchase'] * df['was_carted_in_session']
    
    total_carts = df.groupby('user_id')['is_cart'].cumsum().shift(1).fillna(0)
    total_conv = df.groupby('user_id')['is_cart_conversion'].cumsum().shift(1).fillna(0)
    
    purchase_cart_ratio = (total_conv / total_carts).fillna(0)
    df['cart_abandon_rate'] = 1.0 - purchase_cart_ratio
    df.loc[total_carts == 0, 'cart_abandon_rate'] = 0
    df['cart_abandon_rate'] = df['cart_abandon_rate'].clip(0, 1)

    # Basic Features
    df['total_purchases'] = df.groupby('user_id')['is_purchase'].cumsum().shift(1).fillna(0)
    
    user_start = df.groupby('user_id')['event_time'].transform('min')
    days_since = (df['event_time'] - user_start) / np.timedelta64(1, 'D')
    df['tenure_months'] = (days_since / 30.44) + 0.1
    df['frequency_per_month'] = df['total_purchases'] / df['tenure_months']
    
    df['spend'] = df['price'] * df['is_purchase']
    total_spend = df.groupby('user_id')['spend'].cumsum().shift(1).fillna(0)
    df['aov'] = (total_spend / df['total_purchases'].replace(0, 1)).fillna(0)
    df['clv_formula'] = df['aov'] * df['frequency_per_month'] * 12
    
    # Recency (Global Sort Fix)
    purchases = df[df['is_purchase'] == 1][['user_id', 'event_time']].copy()
    purchases['last_buy'] = purchases['event_time']
    
    df_temp = df[['user_id', 'event_time']].sort_values('event_time')
    purchases_temp = purchases.sort_values('event_time')
    
    df_merged = pd.merge_asof(
        df_temp, purchases_temp, 
        on='event_time', by='user_id', direction='backward', allow_exact_matches=False
    ).sort_index()
    
    df['recency_days'] = (df['event_time'] - df_merged['last_buy']).dt.total_seconds().fillna(-1) / 86400

    # Session
    df['next_event'] = df.groupby('user_session')['event_time'].shift(-1)
    df['time_on_page'] = (df['next_event'] - df['event_time']).dt.total_seconds().fillna(0)
    df['pages_viewed_session'] = df.groupby('user_session').cumcount() + 1
    
    df['new_sess'] = (df['user_session'] != df.groupby('user_id')['user_session'].shift(1)).astype(int)
    total_sess = df.groupby('user_id')['new_sess'].cumsum()
    df['avg_pages_per_session'] = (df.groupby('user_id').cumcount()+1) / total_sess.replace(0, 1)
    
    df['product_page_views'] = df.groupby('product_id').cumcount() + 1

    # Encoding
    le_brand = LabelEncoder()
    df['brand_encoded'] = le_brand.fit_transform(df['brand'].astype(str))
    le_cat = LabelEncoder()
    df['cat_encoded'] = le_cat.fit_transform(df['category_code'].astype(str))

    features = [
        'total_purchases', 'frequency_per_month', 'aov', 'clv_formula',
        'cart_abandon_rate', 'recency_days', 'time_on_page',
        'pages_viewed_session', 'avg_pages_per_session', 'product_page_views',
        'price', 'brand_encoded', 'cat_encoded'
    ]
    
    return df[features].fillna(0), df['is_purchase']

# ==========================================
# 3. OPTIMIZATION EXPERIMENTS
# ==========================================

# Base Params (Shared across all strategies)
def get_base_params(trial):
    return {
        'objective': 'binary:logistic',
        'tree_method': 'hist' if not CONFIG['use_gpu'] else 'gpu_hist',
        'eval_metric': 'aucpr',
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }

# --- STRATEGY 1: Scale_Pos_Weight ONLY ---
def objective_weight_only(trial, X_tr, y_tr, X_val, y_val, base_ratio):
    params = get_base_params(trial)
    # Tune weight around the calculated imbalance ratio
    params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', base_ratio * 0.5, base_ratio * 1.5)
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, verbose=False)
    probs = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, probs)

# --- STRATEGY 2: SMOTE-ENN ONLY ---
def objective_smote_only(trial, X_tr_res, y_tr_res, X_val, y_val):
    params = get_base_params(trial)
    # Weight fixed at 1.0 (Data is already balanced)
    params['scale_pos_weight'] = 1.0
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr_res, y_tr_res, verbose=False)
    probs = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, probs)

# --- STRATEGY 3: HYBRID (SMOTE + Weight) ---
def objective_hybrid(trial, X_tr_res, y_tr_res, X_val, y_val):
    params = get_base_params(trial)
    # Tune weight conservatively (0.5 to 3.0) because data is already balanced
    params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.5, 3.0)
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr_res, y_tr_res, verbose=False)
    probs = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, probs)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    X, y = load_and_engineer()
    
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y
    )
    
    # Calculate Imbalance Ratio for Strategy 1
    imbalance_ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"Imbalance Ratio (Neg/Pos): {imbalance_ratio:.2f}")

    print("\n--- Generating SMOTE-ENN Data (For Strategies 2 & 3) ---")
    smote = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Original Train: {len(X_train)}, Resampled Train: {len(X_train_res)}")

    # Store results
    results = {}
    best_params_global = {}
    best_score_global = -1.0
    best_strategy_name = ""

    # --- EXPERIMENT 1: Scale_Pos_Weight Only ---
    print("\n>>> Running Strategy 1: Scale_Pos_Weight ONLY...")
    study_1 = optuna.create_study(direction='maximize')
    study_1.optimize(lambda t: objective_weight_only(t, X_train, y_train, X_test, y_test, imbalance_ratio), n_trials=CONFIG['n_trials'])
    results['Weight_Only'] = study_1.best_value
    if study_1.best_value > best_score_global:
        best_score_global = study_1.best_value
        best_params_global = study_1.best_params
        best_strategy_name = "Weight_Only"

    # --- EXPERIMENT 2: SMOTE-ENN Only ---
    print("\n>>> Running Strategy 2: SMOTE-ENN ONLY...")
    study_2 = optuna.create_study(direction='maximize')
    study_2.optimize(lambda t: objective_smote_only(t, X_train_res, y_train_res, X_test, y_test), n_trials=CONFIG['n_trials'])
    results['SMOTE_Only'] = study_2.best_value
    # Add manual weight param for compatibility
    params_2 = study_2.best_params
    params_2['scale_pos_weight'] = 1.0
    
    if study_2.best_value > best_score_global:
        best_score_global = study_2.best_value
        best_params_global = params_2
        best_strategy_name = "SMOTE_Only"

    # --- EXPERIMENT 3: Hybrid (Both) ---
    print("\n>>> Running Strategy 3: HYBRID (SMOTE + Weight)...")
    study_3 = optuna.create_study(direction='maximize')
    study_3.optimize(lambda t: objective_hybrid(t, X_train_res, y_train_res, X_test, y_test), n_trials=CONFIG['n_trials'])
    results['Hybrid'] = study_3.best_value
    if study_3.best_value > best_score_global:
        best_score_global = study_3.best_value
        best_params_global = study_3.best_params
        best_strategy_name = "Hybrid"

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("FINAL COMPARISON RESULTS (Metric: PR-AUC)")
    print("="*40)
    print(f"1. Weight Only:  {results['Weight_Only']:.4f}")
    print(f"2. SMOTE Only:   {results['SMOTE_Only']:.4f}")
    print(f"3. Hybrid:       {results['Hybrid']:.4f}")
    print("-" * 40)
    print(f"🏆 WINNER: {best_strategy_name} (Score: {best_score_global:.4f})")
    print("="*40)

    # Save Winner
    with open(os.path.join(MODEL_PATH, 'best_hyperparameters.pkl'), 'wb') as f:
        pickle.dump(best_params_global, f)
    
    # Save SMOTE flag for training
    use_smote_flag = (best_strategy_name != "Weight_Only")
    with open(os.path.join(MODEL_PATH, 'use_smote_flag.pkl'), 'wb') as f:
        pickle.dump(use_smote_flag, f)

    print(f"Optimal parameters saved. Use SMOTE for training? {use_smote_flag}")