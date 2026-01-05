import pandas as pd
import pickle
import os
import gc  # Garbage Collector interface
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, fpgrowth

# ==========================================
# 1. CONFIGURATION
# ==========================================
MIN_SUPPORT = 0.00008      # 0.008% occurrence rate
MIN_CONFIDENCE = 0.02    # 2% certainty
MAX_PRODUCTS = 60000  # Safety Limit

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: adjusted path to look 1 folder up from 'Model' to find 'Data'
DATA_PATH = os.path.join(BASE_DIR, '..', 'Data', 'data.csv') 
MODEL_DIR = os.path.join(BASE_DIR)

def retrain_fpgrowth_safe():
    print(f"Loading Data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. FILTER TRANSACTIONS
    transactions_df = df[df['event_type'].isin(['purchase', 'cart'])]
    print(f"Total Interactions: {len(transactions_df)}")
    
    # 2. SAFETY PRUNING
    top_products = transactions_df['product_id'].value_counts().head(MAX_PRODUCTS).index
    
    print(f"⚠️ CATALOG SIZE WARNING: {len(transactions_df['product_id'].unique())} unique items detected.")
    print(f"📉 PRUNING: Keeping only the Top {MAX_PRODUCTS} most active items.")
    
    transactions_df = transactions_df[transactions_df['product_id'].isin(top_products)]
    
    # 3. CREATE BASKETS
    print("Grouping baskets...")
    baskets = transactions_df.groupby('user_id')['product_id'].apply(list).values.tolist()
    print(f"Analyzing {len(baskets)} user baskets.")
    
    del df
    del transactions_df
    gc.collect()

    # 4. ENCODING (Sparse Matrix Mode)
    print("Encoding transactions (Sparse Mode)...")
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets, sparse=True)
    df_encoded = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    
    # --- FIX IS HERE ---
    # Convert integer Product IDs (columns) to Strings to satisfy mlxtend requirements
    df_encoded.columns = df_encoded.columns.astype(str)
    # -------------------

    # 5. FP-GROWTH
    print(f"Running FP-Growth (Min Support: {MIN_SUPPORT})...")
    frequent_itemsets = fpgrowth(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("❌ ERROR: No itemsets found. Try lowering MIN_SUPPORT.")
        return

    print(f"Found {len(frequent_itemsets)} frequent itemsets.")

    # 6. RULES
    print(f"Generating Rules (Min Conf: {MIN_CONFIDENCE})...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    
    # Convert antecedents/consequents back to integers for the dashboard to use easily
    # (Since we converted them to strings in Step 4, we revert them here)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: frozenset([int(i) for i in x]))
    rules['consequents'] = rules['consequents'].apply(lambda x: frozenset([int(i) for i in x]))

    rules = rules[rules['lift'] > 1.0]
    rules = rules.sort_values('lift', ascending=False)
    
    print(f"✅ Generated {len(rules)} rules.")
    
    # 7. SAVE
    output_path = os.path.join(MODEL_DIR, 'fpgrowth_rules.pkl')
    pd.to_pickle(rules, output_path)
    print(f"Rules saved to {output_path}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    retrain_fpgrowth_safe()