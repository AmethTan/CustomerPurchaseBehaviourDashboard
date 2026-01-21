import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import pickle
import os
import random
import heapq
import altair as alt

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Initialize Streamlit page settings
st.set_page_config(
    page_title="E-commerce CusPurBeAD",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VISUAL CONSTANTS ---
# Threshold for determining if a probability counts as a "Buy" prediction visually (Green vs Red)
VISUAL_THRESH = 0.75

# --- PATH SETUP ---
# dynamically determine paths to ensure it runs regardless of where the script is executed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..')) 
MODEL_DIR = os.path.join(ROOT_DIR, 'Model')
DATA_PATH = os.path.join(ROOT_DIR, 'Data', 'data.csv')

# --- CSS Styling ---
# Custom HTML/CSS to render the flowcharts in the Simulation section
st.markdown("""
<style>
    .node-box {
        border: 1px solid #d0d0d0;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        display: inline-block;
        text-align: center;
        font-size: 0.9em;
        min-width: 80px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .node-start { background-color: #ffd700; font-weight: bold; }
    .node-hist { background-color: #e2e3e5; color: #666; }
    .node-future { background-color: #dbeafe; color: #1e3a8a; }
    .arrow {
        font-size: 1.5em;
        vertical-align: middle;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================
# PyTorch Definition for the Self-Attentive Sequential Recommendation (SASRec) model
class SASRecModel(nn.Module):
    def __init__(self, num_items, maxlen=49, hidden_units=64, dropout_rate=0.1):
        super(SASRecModel, self).__init__()
        self.item_emb = nn.Embedding(num_items, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        # Transformer Encoder to capture sequential patterns
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units, nhead=2, dim_feedforward=hidden_units*2, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear(hidden_units, num_items)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.item_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        src_mask = (x == 0).all(dim=-1)
        x = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        return self.out(x)

# ==========================================
# 3. RESOURCE LOADING
# ==========================================
@st.cache_resource
def load_resources():
    """
    Loads all heavy assets (Models, Dataframes, Encoders) once and caches them 
    to prevent reloading on every interaction.
    """
    # A. Load Data from GitHub (for deployment compatibility)
    try:
        # Download from raw GitHub content URL
        data_url = "https://media.githubusercontent.com/media/AmethTan/CustomerPurchaseBehaviourDashboard/master/Data/data.csv"
        df = pd.read_csv(data_url)
        st.success("✅ Data loaded from GitHub")
        
    except Exception as e:
        # Fallback: try loading locally if available (for local development)
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            st.info("📂 Data loaded locally")
        else:
            # Fallback empty structure to prevent crashes if file missing
            st.error(f"Failed to load data: {e}")
            df = pd.DataFrame(columns=['event_time', 'event_type', 'product_id', 'category_code', 'brand', 'price', 'user_session'])

    # B. Load XGBoost (Purchase Probability)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, 'xgboost_retrained.json'))
    
    # Load metadata for XGBoost
    with open(os.path.join(MODEL_DIR, 'xgb_threshold.pkl'), 'rb') as f: xgb_threshold = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f: xgb_features = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'le_brand_retrained.pkl'), 'rb') as f: le_brand = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'le_cat_retrained.pkl'), 'rb') as f: le_cat = pickle.load(f)
    
    # C. Load FP-Growth (Association Rules/Bundles)
    fp_rules = pd.read_pickle(os.path.join(MODEL_DIR, 'fpgrowth_rules.pkl'))
    
    # D. Load SASRec (Next Click Prediction)
    with open(os.path.join(MODEL_DIR, 'item_map.pkl'), 'rb') as f: item_map = pickle.load(f)
    inv_item_map = {v: k for k, v in item_map.items()}
    
    sas_model = SASRecModel(num_items=len(item_map)+1)
    sas_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'sasrec_model.pth'), map_location='cpu'))
    sas_model.eval()

    # Create Product Lookup Table (Fast access to product details by ID)
    view_counts = df['product_id'].value_counts()
    product_lookup = df.sort_values('event_time').groupby('product_id').last()[['category_code', 'brand', 'price']]
    product_lookup['views'] = product_lookup.index.map(view_counts)
    product_lookup['brand'] = product_lookup['brand'].fillna('UNKNOWN')
    product_lookup['category_code'] = product_lookup['category_code'].fillna('unknown')
    
    # Pre-calculate unique lists for Advanced Search dropdowns
    unique_cats = sorted(product_lookup['category_code'].astype(str).unique().tolist())
    unique_brands = sorted(product_lookup['brand'].astype(str).unique().tolist())
    
    return {
        'xgb': xgb_model, 'xgb_thresh': xgb_threshold, 'xgb_feats': xgb_features,
        'le_brand': le_brand, 'le_cat': le_cat,
        'fp_rules': fp_rules,
        'sas_model': sas_model, 'item_map': item_map, 'inv_item_map': inv_item_map,
        'product_lookup': product_lookup,
        'unique_cats': unique_cats,
        'unique_brands': unique_brands
    }

resources = load_resources()

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def predict_purchase_prob(customer_data, product_data):
    """
    Encodes input data and runs the XGBoost model to predict 
    the probability of a 'purchase' event.
    """
    input_data = {**customer_data, **product_data}
    
    # Handle label encoding safely (handle unseen brands/categories)
    def safe_encode(encoder, val):
        val = str(val)
        return encoder.transform([val])[0] if val in encoder.classes_ else -1
        
    input_data['brand_encoded'] = safe_encode(resources['le_brand'], input_data['brand'])
    input_data['cat_encoded'] = safe_encode(resources['le_cat'], input_data['category_code'])
    
    # Ensure columns are in the exact order the model expects
    df_input = pd.DataFrame([input_data])[resources['xgb_feats']]
    prob = resources['xgb'].predict_proba(df_input)[:, 1][0]
    return prob

def get_sasrec_predictions(sequence, top_k=100, allow_repeats=False):
    """
    Infers the next likely items based on the user's clickstream sequence 
    using the loaded PyTorch SASRec model.
    """
    if not sequence: return []
    
    current_item = sequence[-1] if sequence else None
    
    # Convert Product IDs to Model Index IDs
    seq_ids = [resources['item_map'].get(x, 0) for x in sequence]
    maxlen = 49
    # Pad sequence to match model input size
    seq_ids = ([0] * (maxlen - len(seq_ids))) + seq_ids[-maxlen:]
    
    with torch.no_grad():
        logits = resources['sas_model'](torch.LongTensor([seq_ids]))
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        # Fetch top K probabilities
        fetch_k = top_k + 5 if not allow_repeats else top_k
        top_probs, top_indices = torch.topk(probs, fetch_k)
        
    results = []
    for i in range(len(top_indices[0])):
        idx = top_indices[0][i].item()
        if idx == 0: continue # Skip padding
        pid = resources['inv_item_map'].get(idx)
        
        # Logic to prevent suggesting the item the user is currently looking at (unless echo mode is on)
        if not allow_repeats and pid and str(pid) == str(current_item):
            continue
            
        if pid:
            results.append({'product_id': pid, 'sas_prob': top_probs[0][i].item()})
            
    return results[:top_k]

def beam_search_clickstreams(start_sequence, depth=10, width=3, allow_repeats=False):
    """
    Simulates future user paths using Beam Search algorithm.
    It explores the 'width' most likely paths at each 'depth' step.
    """
    queue = [(0.0, start_sequence)] 
    completed_paths = []
    
    for _ in range(depth):
        candidates = []
        while queue:
            score, seq = heapq.heappop(queue)
            # Get next step possibilities
            next_items = get_sasrec_predictions(seq, top_k=width, allow_repeats=allow_repeats)
            
            for item in next_items:
                new_seq = seq + [item['product_id']]
                # Accumulate Negative Log Likelihood (Score)
                new_score = score - np.log(item['sas_prob'] + 1e-9) 
                candidates.append((new_score, new_seq))
        
        if not candidates: break
        candidates.sort(key=lambda x: x[0]) 
        # Keep only the top 'width' candidates for the next iteration
        queue = candidates[:width]
        for c in queue:
            completed_paths.append(c)

    results = []
    for score, seq in completed_paths:
        # Convert Log Score back to Probability
        chain_prob = np.exp(-score)
        future_part = seq[len(start_sequence):]
        if future_part:
            results.append({'chain': future_part, 'prob': chain_prob, 'full_seq': seq})
    
    return sorted(results, key=lambda x: x['prob'], reverse=True)[:10]

def enrich_and_calc(pool, source_name, customer_data):
    """
    Takes a list of product recommendations, looks up their details (Price, Brand),
    and calculates the Purchase Probability for each using XGBoost.
    """
    if not pool: return pd.DataFrame()
    df = pd.DataFrame(pool).drop_duplicates('product_id')
    
    def get_info(pid):
        try:
            r = resources['product_lookup'].loc[pid]
            return r['price'], r['category_code'], r['brand']
        except: return 0.0, 'unknown', 'unknown'
        
    df[['price', 'category', 'brand']] = df['product_id'].apply(lambda x: pd.Series(get_info(x)))
    
    def calc_prob(row):
        # Predict probability assuming user is viewing THIS recommended product
        return predict_purchase_prob(customer_data, {
            'category_code': row['category'], 'brand': row['brand'], 
            'price': row['price'], 'product_page_views': 100
        })
    df['xgb_buy_prob'] = df.apply(calc_prob, axis=1)
    df['joint_prob'] = df['score'] * df['xgb_buy_prob']
    df['source'] = source_name
    return df

# ==========================================
# 5. STATE & CALLBACKS
# ==========================================
# Initialize Session State variables
if 'clickstream' not in st.session_state: st.session_state['clickstream'] = []
if 'search_query' not in st.session_state: st.session_state['search_query'] = ""
if 'echo_mode' not in st.session_state: st.session_state['echo_mode'] = False
if 'inspected_product' not in st.session_state: st.session_state['inspected_product'] = None
if 'search_total' not in st.session_state: st.session_state['search_total'] = 0

# --- Customer Callbacks ---
def update_clv():
    # Simple CLV formula: AOV * Frequency * 12 months
    st.session_state['c_clv'] = st.session_state['c_aov'] * st.session_state['c_freq'] * 12

def clear_components():
    st.session_state['c_freq'] = 0.0
    st.session_state['c_aov'] = 0.0

# --- Product Callbacks ---
def on_search_submit():
    """Triggered when user hits Enter on search box or confirms selection"""
    query = st.session_state['search_query']
    if query:
        try:
            pid = int(query)
            if pid in resources['product_lookup'].index:
                # Fill sidebar fields with found product data
                row = resources['product_lookup'].loc[pid]
                st.session_state['p_cat'] = str(row['category_code'])
                st.session_state['p_brand'] = str(row['brand'])
                st.session_state['p_price'] = float(row['price'])
                st.session_state['p_views'] = int(row['views'])
                st.session_state['inspected_product'] = pid # Auto-inspect on search
            else:
                st.toast("Product ID not found.", icon="⚠️")
        except ValueError:
            st.toast("Please enter a numeric ID.", icon="⚠️")

def on_field_change():
    # If user manually changes sidebar fields, clear specific search ID
    st.session_state['search_query'] = ""

# ==========================================
# 6. SIDEBAR INPUTS & SETUP
# ==========================================
st.sidebar.title("🛠️ Analysis Controls")

# --- DOCUMENTATION & CREDITS ---
with st.sidebar.expander("📘 Dashboard Guide & Credits", expanded=False):
    st.markdown("""
    **Dashboard Description**
    \n This application helps e-commerce managers predict customer purchase probability, suggesting product bundles and simulate future browsing paths using machine learning models (XGBoost, FP-Growth & SASRec).

    **Input Variables - Customer:**
    * **Recency (Days):** Days since the customer's last visit.
    * **Freq/Month:** Average number of purchases per month.
    * **AOV ($):** Average Order Value.
    * **CLV ($):** Customer Lifetime Value (Total expected revenue).
    * **Time/Page (s):** Average time spent on a product page.
    * **Abandon Rate:** Average cart abandonment rate of the customer.
    * **Pages Viewed:** Total pages viewed in the session.
    * **Avg Pages/Session:** Average pages viewed per session.

    **Input Variables - Product:**
    * **Category Code:** The hierarchical category (e.g., electronics.smartphone).
    * **Brand:** The manufacturer of the product.
    * **Price ($):** The selling price of the item.
    * **Popularity:** Estimated views (product page impressions).

    **How to Use:**
    1.  **Purchase Prediction:** Adjust "Customer Profile" and "Product Interaction" inputs.
    2.  **Smart Recommendations:** View products frequently bought together or predicted next clicks.
    3.  **Simulation:** Add products to "Session History" to simulate user paths.
    \n (Note: Select products from "Smart Recommendations" and "Simulation" in the **Product Inspector** below to view details.)

    **Dataset Credits:**
    This work uses the **"eCommerce events history in electronics store"** dataset (Oct 2019 – Feb 2020).
    * **Source:** [REES46 Marketing Platform](https://rees46.com/)
    * **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store)
    """)

# --- Customer Profile Inputs ---
st.sidebar.header("👤 Customer Profile")
with st.sidebar.expander("Metrics", expanded=True):
    st.slider("Recency (Days)", 0, 365, 30, key='c_recency')
    c1, c2 = st.columns(2)
    c1.number_input("Freq/Month", 0.0, None, 1.5, 0.1, key='c_freq', on_change=update_clv)
    c2.number_input("AOV ($)", 0.0, None, 250.0, 10.0, key='c_aov', on_change=update_clv)
    st.number_input("CLV ($)", 0.0, None, 4500.0, 100.0, key='c_clv', on_change=clear_components)
    st.slider("Time/Page (s)", 0, 1800, 45, key='c_time_page')
    st.slider("Abandon Rate", 0.0, 1.0, 0.5, key='c_abandon')
    st.number_input("Pages Viewed", 1, None, 5, key='c_views')
    st.number_input("Avg Pages/Session", 1.0, None, 8.5, key='c_avg_views')

    # Dictionary used for XGBoost inference
    customer_data = {
        'recency_days': st.session_state['c_recency'], 'frequency_per_month': st.session_state['c_freq'],
        'aov': st.session_state['c_aov'], 'clv_formula': st.session_state['c_clv'], 
        'time_on_page': st.session_state['c_time_page'], 'cart_abandon_rate': st.session_state['c_abandon'], 
        'pages_viewed_session': st.session_state['c_views'], 'avg_pages_per_session': st.session_state['c_avg_views'], 
        'total_purchases': st.session_state['c_freq'] * 12 
    }

# --- Product Interaction Inputs ---
st.sidebar.header("📦 Product Interaction")

# --- Search Dialog Function ---
@st.dialog("Advanced Product Search")
def advanced_search_dialog():
    st.markdown("Filter items to find specific IDs.")
    
    # Filter Controls
    col1, col2 = st.columns(2)
    with col1: s_cat = st.selectbox("Category", ["All"] + resources['unique_cats'])
    with col2: s_brand = st.selectbox("Brand", ["All"] + resources['unique_brands'])
    
    col3, col4 = st.columns(2)
    min_p, max_p = int(resources['product_lookup']['price'].min()), int(resources['product_lookup']['price'].max())
    min_v, max_v = int(resources['product_lookup']['views'].min()), int(resources['product_lookup']['views'].max())
    with col3: s_price = st.slider("Price Range ($)", min_p, max_p, (min_p, max_p))
    with col4: s_views = st.slider("Popularity (Views)", min_v, max_v, (min_v, max_v))
        
    if st.button("🔍 Search"):
        # Apply filters to the lookup dataframe
        df = resources['product_lookup'].copy()
        if s_cat != "All": df = df[df['category_code'] == s_cat]
        if s_brand != "All": df = df[df['brand'] == s_brand]
        df = df[(df['price'] >= s_price[0]) & (df['price'] <= s_price[1])]
        df = df[(df['views'] >= s_views[0]) & (df['views'] <= s_views[1])]
        
        # INCREASED LIMIT TO 1000 to improve searchability while maintaining performance
        count_total = len(df)
        st.session_state['search_total'] = count_total
        st.session_state['search_results'] = df.head(1000) 
        
    if 'search_results' in st.session_state:
        if st.session_state['search_results'].empty:
            st.warning("No product fulfills the constraints.")
        else:
            # Display Count and Warning if truncated
            res_len = len(st.session_state['search_results'])
            total = st.session_state.get('search_total', res_len)
            
            if total > res_len:
                st.warning(f"Found {total} items. Showing top {res_len}. Please refine filters.")
            else:
                st.write(f"Found {total} items.")
            
            # 1. Reset Index to get product_id as a column
            display_df = st.session_state['search_results'].reset_index()

            # 2. RENAME COLUMNS for better User Experience
            display_df = display_df.rename(columns={
                "product_id": "Product ID",
                "category_code": "Category",
                "brand": "Brand",
                "price": "Price ($)",
                "views": "Total Views"
            })
            
            # 3. Display with new names using width='stretch'
            st.dataframe(display_df, hide_index=True, width='stretch')
            
            # 4. Update Selectbox to use the new 'Product ID' column name
            selected_id = st.selectbox("Select Product from List:", display_df['Product ID'].tolist())
            
            if st.button("✅ Confirm Selection"):
                st.session_state['search_query'] = str(selected_id)
                on_search_submit()
                st.rerun()

col_search, col_adv = st.sidebar.columns([3, 1])
with col_search:
    st.text_input("Search ID", key='search_query', on_change=on_search_submit, label_visibility="collapsed", placeholder="Prod ID")
with col_adv:
    if st.button("🔎", help="Advanced Search"): advanced_search_dialog()

# Default values if not set
if 'p_cat' not in st.session_state: st.session_state['p_cat'] = "electronics.smartphone"
if 'p_brand' not in st.session_state: st.session_state['p_brand'] = "SAMSUNG"
if 'p_price' not in st.session_state: st.session_state['p_price'] = 500.0
if 'p_views' not in st.session_state: st.session_state['p_views'] = 100

st.sidebar.text_input("Category Code", key='p_cat', on_change=on_field_change)
st.sidebar.text_input("Brand", key='p_brand', on_change=on_field_change)
st.sidebar.number_input("Price ($)", 0.0, None, 500.0, 10.0, key='p_price', on_change=on_field_change)
st.sidebar.number_input("Popularity", 0, None, 100, 1, key='p_views', on_change=on_field_change)

product_data = {
    'category_code': st.session_state['p_cat'], 'brand': st.session_state['p_brand'].upper(), 
    'price': st.session_state['p_price'], 'product_page_views': st.session_state['p_views']
}

# Add current search ID to the simulation clickstream
if st.sidebar.button("➕ Add to Clickstream"):
    cs = st.session_state['search_query']
    if cs and cs.isdigit() and int(cs) in resources['product_lookup'].index:
        st.session_state['clickstream'].append(int(cs))
        st.sidebar.success(f"Added {cs}")
    else:
        st.sidebar.warning("Search valid ID first.")

# Toggle to allow repeating items in next-click prediction
st.sidebar.checkbox("Allow Repeated Items (Echo)", key='echo_mode')

# History Display
st.sidebar.markdown("### 📜 Session History")
if st.session_state['clickstream']:
    st.sidebar.caption(" → ".join([str(x) for x in st.session_state['clickstream'][-5:]]))
    if st.sidebar.button("Clear History"):
        st.session_state['clickstream'] = []
        st.rerun()
else:
    st.sidebar.info("Empty")

# ==========================================
# 7. PRE-CALCULATION FOR UNIFIED INSPECTOR
# ==========================================
# We run light inference here to collect all product IDs that appear 
# in the main dashboard (Bundles, Predictions, Simulation) to populate the sidebar dropdown.
inspect_candidates = set()
if st.session_state['clickstream']: inspect_candidates.update(st.session_state['clickstream'])
if st.session_state.get('search_query'): 
    try: inspect_candidates.add(int(st.session_state['search_query']))
    except: pass

# 1. Gather Bundles Candidates
recs_fp = []
curr_s = st.session_state['search_query']
if curr_s and curr_s.isdigit():
    pid_int = int(curr_s)
    rels = resources['fp_rules'][resources['fp_rules']['antecedents'].apply(lambda x: pid_int in x)].head(5)
    for _, r in rels.iterrows(): inspect_candidates.add(list(r['consequents'])[0])

# 2. Gather SASRec Candidates
recs_sas = get_sasrec_predictions(st.session_state['clickstream'], 5, st.session_state['echo_mode'])
for r in recs_sas: inspect_candidates.add(r['product_id'])

# 3. Gather Simulation Candidates (Top path only)
if st.session_state['clickstream']:
    sim_paths = beam_search_clickstreams(st.session_state['clickstream'], allow_repeats=st.session_state['echo_mode'])
    if sim_paths:
        for pid in sim_paths[0]['chain']: inspect_candidates.add(pid)

# --- UNIFIED INSPECTOR WIDGET ---
st.sidebar.divider()
st.sidebar.header("🔍 Product Inspector")
inspect_list = sorted(list(inspect_candidates))

if inspect_list:
    # Use index to keep selection consistent if possible
    idx = 0
    if st.session_state['inspected_product'] in inspect_list:
        idx = inspect_list.index(st.session_state['inspected_product'])
    
    selected_inspect = st.sidebar.selectbox("Select Product to View Details:", inspect_list, index=idx)
    st.session_state['inspected_product'] = selected_inspect
    
    # Render Details Panel
    if selected_inspect and selected_inspect in resources['product_lookup'].index:
        p_row = resources['product_lookup'].loc[selected_inspect]
        with st.sidebar.container(border=True):
            st.info(f"**Product ID:** {selected_inspect}")
            st.write(f"**Brand:** {p_row['brand']}")
            st.write(f"**Category:** {p_row['category_code']}")
            st.write(f"**Price:** ${p_row['price']:.2f}")
            st.write(f"**Views:** {p_row['views']}")
else:
    st.sidebar.caption("Interact with search, recommendations, or simulation to see products here.")


# ==========================================
# 8. DASHBOARD MAIN
# ==========================================
st.title("🛍️ E-commerce CusPurBeAD")
st.markdown("Predictive & Prescriptive Analytics for Online Sales")
st.divider()

# --- 1. PURCHASE PREDICTION ---
st.subheader("1. Purchase Prediction (Current View)")
c1, c2, c3 = st.columns([1, 2, 1])
curr_prob = predict_purchase_prob(customer_data, product_data)
is_buy = curr_prob >= VISUAL_THRESH 

with c1:
    st.markdown("### Likelihood")
    st.metric("Probability", f"{curr_prob:.2%}", "High" if is_buy else "Low")
with c2:
    # Gauge Chart using Altair
    df_chart = pd.DataFrame({'Probability': [curr_prob], 'Label': ['Purchase']})
    base = alt.Chart(df_chart).encode(x=alt.X('Probability', scale=alt.Scale(domain=[0, 1])))
    bar = base.mark_bar(size=40, color='green' if is_buy else 'red').encode()
    text = base.mark_text(align='center', baseline='middle', color='white').encode(text=alt.Text('Probability', format='.1%'))
    st.altair_chart(bar + text, width='stretch')
with c3:
    if is_buy: st.success("✅ Prediction: **BUY**")
    else: st.error("❌ Prediction: **NO BUY**")
    st.caption(f"Visual Threshold: {VISUAL_THRESH:.2f}")

st.divider()

# --- 2. RECOMMENDATIONS ---
st.subheader("2. Smart Recommendations & Next Clicks")
tab1, tab2 = st.tabs(["📦 Product Bundles (Cross-Sell)", "⏭️ Next Click Prediction"])

# Re-run logic for display (full lists)
recs_pool_fp = []
if curr_s and curr_s.isdigit():
    pid_int = int(curr_s)
    # Find rules where current product is the antecedent
    rels = resources['fp_rules'][
        resources['fp_rules']['antecedents'].apply(lambda x: pid_int in x)
    ].sort_values('lift', ascending=False).head(20)
    for _, r in rels.iterrows():
        recs_pool_fp.append({'product_id': list(r['consequents'])[0], 'score': r['confidence']})

recs_pool_sas = []
sas_res = get_sasrec_predictions(st.session_state['clickstream'], 50, st.session_state['echo_mode'])
for r in sas_res:
    recs_pool_sas.append({'product_id': r['product_id'], 'score': r['sas_prob']})

# --- TAB 1: BUNDLES ---
with tab1:
    df_fp = enrich_and_calc(recs_pool_fp, 'FP-Growth', customer_data)
    if not df_fp.empty:
        st.caption("Commonly bought together.")
        cols = ['product_id', 'category', 'brand', 'price', 'xgb_buy_prob', 'score']
        rn = {'xgb_buy_prob': 'Purchase Odds', 'score': 'Confidence', 'product_id': 'Product ID', 'price': 'Price ($)', 'category': 'Category', 'brand': 'Brand'}
        st.dataframe(
            df_fp[cols].sort_values('xgb_buy_prob', ascending=False).head(100).rename(columns=rn)
            .style.format({'Price ($)': '{:.2f}', 'Purchase Odds': '{:.2%}', 'Confidence': '{:.2%}'}),
            width='stretch', hide_index=True
        )
    else:
        st.info("No bundles found. Try searching for a product with associations.")

# --- TAB 2: NEXT CLICKS ---
with tab2:
    df_sas = enrich_and_calc(recs_pool_sas, 'SASRec', customer_data)
    if not df_sas.empty:
        st.caption(f"Predicted next interests.")
        cols = ['product_id', 'category', 'score', 'xgb_buy_prob', 'joint_prob']
        rn = {'score': 'Next Click Odds', 'xgb_buy_prob': 'Purchase Odds', 'joint_prob': 'Joint Odds', 'product_id': 'Product ID', 'category': 'Category', 'brand': 'Brand'}
        st.dataframe(
            df_sas[cols].sort_values('score', ascending=False).head(100).rename(columns=rn)
            .style.format({'Next Click Odds': '{:.2%}', 'Purchase Odds': '{:.2%}', 'Joint Odds': '{:.2%}'}),
            width='stretch', hide_index=True
        )
    else:
        st.info("No predictions. Add items to history.")

st.divider()

# --- 3. SIMULATION ---
st.subheader("3. Future Clickstream Simulation")
st.markdown("Simulating user journey.")

if st.session_state['clickstream']:
    # Re-run beam search for display
    paths = beam_search_clickstreams(st.session_state['clickstream'], allow_repeats=st.session_state['echo_mode'])
    
    e_paths = []
    for p in paths:
        probs, prices = [], []
        # Calculate expected value for each path
        for pid in p['chain']:
            try:
                row = resources['product_lookup'].loc[pid]
                pr, cat, br = row['price'], row['category_code'], row['brand']
            except: pr, cat, br = 0, 'unknown', 'unknown'
            
            pb = predict_purchase_prob(customer_data, {'category_code': cat, 'brand': br, 'price': pr, 'product_page_views': 100})
            probs.append(pb)
            prices.append(pr)
        
        p['avg_buy_prob'] = np.mean(probs) if probs else 0
        p['expected_value'] = np.sum(np.array(prices) * np.array(probs))
        e_paths.append(p)
    
    # Sorting Controls
    sort_c = st.selectbox("Sort By:", ["Most Possible Clickstream", "Highest Purchase Odds", "Highest Expected Value"])
    if sort_c == "Most Possible Clickstream": e_paths.sort(key=lambda x: x['prob'], reverse=True)
    elif sort_c == "Highest Purchase Odds": e_paths.sort(key=lambda x: x['avg_buy_prob'], reverse=True)
    else: e_paths.sort(key=lambda x: x['expected_value'], reverse=True)

    # Render Visual Paths
    for p in e_paths[:10]:
        with st.container():
            cv, cm = st.columns([3, 1])
            with cv:
                h = "<div class='node-box node-start'>START</div> <span class='arrow'>&rarr;</span> "
                for pid in st.session_state['clickstream'][-3:]:
                    h += f"<div class='node-box node-hist'>Hist: {pid}</div> <span class='arrow'>&rarr;</span> "
                for pid in p['chain']:
                    try: pr = resources['product_lookup'].loc[pid]['price']
                    except: pr = 0
                    h += f"<div class='node-box node-future'><b>{pid}</b><br>${pr:.0f}</div> <span class='arrow'>&rarr;</span> "
                
                bg = "#d4edda" if p['avg_buy_prob'] >= VISUAL_THRESH else "#f8d7da"
                lbl = "BUY" if p['avg_buy_prob'] >= VISUAL_THRESH else "EXIT"
                h += f"<div class='node-box' style='background-color:{bg};font-weight:bold'>{lbl}</div>"
                st.markdown(h, unsafe_allow_html=True)
            with cm:
                st.caption(f"Path Odds: {p['prob']:.1%}")
                st.caption(f"Expected Value: ${p['expected_value']:.2f}")
                st.caption(f"Avg. Purchase Odds: {p['avg_buy_prob']:.1%}")
            st.divider()
else:
    st.warning("Add items to history to simulate.")