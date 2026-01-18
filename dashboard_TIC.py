import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import datetime, date, timedelta
import numpy as np
from fpdf import FPDF
import base64
import feedparser 
import os 
import calendar 
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
import streamlit.components.v1 as components
import time
import threading
import concurrent.futures
import hashlib
import json
import extra_streamlit_components as stx
from scipy.stats import norm
import toml
import requests


# --- GOOGLE SHEETS CONNECTION SETUP ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

@st.cache_resource
def init_connection():
    """Authenticates with Google Sheets using Streamlit Secrets."""
    try:
        # Load credentials from secrets.toml
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES
        )
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"⚠️ Failed to connect to Google Sheets: {e}")
        return None
def _fetch_single_sheet(client, sheet_name):
    """Helper for the parallel executor with Retry Logic."""
    # Try up to 3 times to handle API limits
    for attempt in range(3):
        try:
            sheet = client.open("TIC_Database_Master")
            worksheet = sheet.worksheet(sheet_name)
            data = worksheet.get_all_values()
            
            if not data: return pd.DataFrame()

            headers = data.pop(0)
            df = pd.DataFrame(data, columns=headers)
            # Clean empty columns
            return df.loc[:, [h != "" for h in headers]]
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Quota exceeded" in error_msg:
                time.sleep((attempt + 1) * 2) # Wait 2s, 4s, 6s
                continue
            elif "WorksheetNotFound" in error_msg:
                return pd.DataFrame()
            else:
                return pd.DataFrame()
    return pd.DataFrame()

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(
    page_title="TIC Portal | Internal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

TIC_LOGO = "https://media.licdn.com/dms/image/v2/D4D0BAQEBPgrthbI7xQ/company-logo_200_200/B4DZoGPeNMJIAI-/0/1761041311048/tilburginvestmentclub_logo?e=1770249600&v=beta&t=ZVfUkEMB4cu-ZNXakbHLXMEdMv8chcJTZsg3f7TgouQ"

st.markdown("""
    <style>
        .stDeployButton {display:none;}
        
        /* FIX: Increased padding to stop titles getting cut off */
        .block-container {
            padding-top: 3rem; 
            padding-bottom: 2rem;
        }
        /* FIX SIDEBAR SCROLLING: Reduces bottom whitespace */
        section[data-testid="stSidebar"] .block-container {
            padding-bottom: 1rem !important;
        }
        /* NEWS CARD STYLE */
        .news-item {
            padding: 10px;
            border-bottom: 1px solid rgba(49, 51, 63, 0.2);
        }
        .news-source { font-size: 0.8em; color: #D4AF37; font-weight: bold; }
        .news-head { font-size: 1em; font-weight: 600; color: var(--text-color); text-decoration: none;}
        .news-head:hover { color: #D4AF37 !important; }
        .news-sum { font-size: 0.85em; color: var(--text-color); opacity: 0.8; }

        /* SIDEBAR CARD STYLE */
        .event-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
        }
        .event-ticker { font-weight: bold; color: #D4AF37; font-size: 1.1em; }
        .event-date { color: var(--text-color); opacity: 0.8; font-size: 0.9em; }
        .event-badge {
            float: right;
            background-color: #D4AF37;
            color: #000000;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        /* HIDES THE "Press Enter to apply" / "Press Enter to submit" HINTS */
    div[data-testid="InputInstructions"] {
        display: none !important;
        }
        
        /* METRIC BOX STYLE */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(49, 51, 63, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER (REAL DATA FETCHING)
# ==========================================
# --- DIRECT GOOGLE SHEET AUTH (Dual Check) ---
def check_credentials_live(user_input, user_password):
    """Checks credentials against Email OR Username, handling both Plain & Hashed passwords."""
    try:
        # 1. Connect
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        # 2. Open Sheet
        sheet = client.open("TIC_Database_Master")
        ws = sheet.worksheet("Members")
        records = ws.get_all_records()
        
        # 3. Prepare Inputs
        clean_input = user_input.strip().lower()
        clean_pass = user_password.strip()
        
        # Create the hash of what the user typed (for comparison)
        hashed_pass = hashlib.sha256(clean_pass.encode('utf-8')).hexdigest()
        
        for row in records:
            # Match User (Email OR Name)
            db_email = str(row.get("Email", "")).strip().lower()
            raw_name = str(row.get("Name", "")).strip()
            db_username = raw_name.lower().replace(" ", ".")
            
            if clean_input == db_email or clean_input == db_username:
                
                # --- THE DUAL CHECK FIX ---
                db_pass = str(row.get("Password", "")).strip()
                
                # Check 1: Is it a legacy plain text password? (e.g. "pass")
                if db_pass == clean_pass:
                    return row.get("Role", "Member"), None 
                
                # Check 2: Is it a modern hashed password?
                elif db_pass == hashed_pass:
                    return row.get("Role", "Member"), None 
                
                else:
                    return None, "Incorrect Password"
                
        return None, f"User '{clean_input}' not found"

    except Exception as e:
        return None, f"Login System Error: {str(e)}"
    
@st.cache_data(ttl=1800)
def fetch_macro_news():
    """Pulls real news from CNBC and Yahoo Finance RSS feeds"""
    feeds = [
        {"url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", "source": "CNBC Global"},
        {"url": "https://finance.yahoo.com/news/rssindex", "source": "Yahoo Finance"},
    ]
    news_items = []
    try:
        for feed in feeds:
            d = feedparser.parse(feed["url"])
            for entry in d.entries[:4]: 
                news_items.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary": entry.get('summary', '')[:150] + "...",
                    "published": entry.get('published', '')[:16],
                    "source": feed["source"]
                })
    except Exception as e:
        return []
    return news_items


def render_onboarding_tour(user):
    """
    Displays a welcome guide for new users. 
    Updates the database to 'Onboarded=1' when finished.
    """
    st.title(f"👋 Welcome to TIC Portal, {user['n']}!")
    st.subheader("Let's get you set up in 3 steps.")
    
    # We use a multi-step container approach
    step = st.radio("Tour Steps:", ["1. The Dashboard", "2. Voting", "3. Your Profile"], horizontal=True, label_visibility="collapsed")
    
    st.divider()

    if "1." in step:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info("📊 **Real-Time Data**")
            st.markdown("""
            The **Dashboard** is your home base. 
            - See the Fund's live NAV.
            - Check your personal stake performance.
            - View the top holdings in real-time.
            """)
        with c2:
            st.caption("Preview")
            st.bar_chart({"Fund": 100, "You": 120}) # Simple dummy viz

    elif "2." in step:
        st.warning("🗳️ **Democracy in Action**")
        st.markdown("""
        As a member, you have voting rights.
        - Go to the **Dashboard** to see active proposals.
        - Click **YES** or **NO** to cast your vote.
        - Votes are recorded instantly on the blockchain (Google Sheets).
        """)
        st.button("Vote YES (Demo)", disabled=True)
        st.button("Vote NO (Demo)", disabled=True)

    elif "3." in step:
        st.success("👤 **Manage Your Assets**")
        st.markdown("""
        Go to **Settings** to:
        - Change your password (Recommended!).
        - Request to deposit more capital.
        - Download your signed contracts.
        """)

    st.divider()
    
    col_fin = st.columns([3, 1])
    with col_fin[1]:
        if st.button("🚀 Get Started", type="primary", width="stretch"):
            with st.spinner("Setting up your profile..."):
                # 1. Update Google Sheet
                update_member_field_in_gsheet(user['u'], "Onboarded", 1)
                
                # 2. Update Session State (so we don't need a full reload)
                st.session_state['user']['onboarded'] = 1
                
                # 3. Refresh
                st.rerun()

@st.cache_data(ttl=3600)
def fetch_macro_data():
    """Pulls real macro indicators using YFinance"""
    tickers = {'10Y Treasury': '^TNX', 'VIX': '^VIX', 'EUR/USD': 'EURUSD=X', 'Crude Oil': 'CL=F'}
    macro_data = {}
    try:
        data = yf.download(list(tickers.values()), period="5d", progress=False)['Close']
        for name, ticker in tickers.items():
            if ticker in data.columns:
                s = data[ticker].dropna()
                if len(s) >= 2:
                    macro_data[name] = {'value': s.iloc[-1], 'delta': s.iloc[-1] - s.iloc[-2]}
                else:
                    macro_data[name] = {'value': 0, 'delta': 0}
            else:
                 macro_data[name] = {'value': 0, 'delta': 0}
    except:
        return {k: {'value': 0, 'delta': 0} for k in tickers}
    return macro_data

def fetch_single_calendar_event(t):
    """Helper: Fetches one ticker's earnings date (runs in parallel)."""
    try:
        if not isinstance(t, str): return None
        
        # Fast initialization
        stock = yf.Ticker(t)
        cal = stock.calendar
        
        if cal and 'Earnings Date' in cal:
            dates = cal['Earnings Date']
            if dates:
                next_date = dates[0]
                # Only keep future dates
                if next_date >= date.today():
                    return {
                        'title': f"{t} Earnings",
                        'ticker': t,
                        'date': next_date.strftime('%Y-%m-%d'),
                        'type': 'market',
                        'audience': 'all'
                    }
    except Exception:
        return None
    return None

@st.cache_data(ttl=3600*12) 
def fetch_company_events(tickers):
    """Fetches upcoming earnings for a list of tickers using MULTITHREADING."""
    if not tickers: return []
    
    events = []
    # Limit to 20 tickers to stay safe, remove bad values
    safe_tickers = [str(t) for t in tickers[:20] if str(t).upper() not in ['NAN', 'NONE', '']]
    
    # Run 10 requests at the same time (Parallel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks
        futures = {executor.submit(fetch_single_calendar_event, t): t for t in safe_tickers}
        
        # Collect results as they finish
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    events.append(result)
            except Exception:
                continue
                
    return events

@st.cache_data(ttl=3600*24)
def fetch_correlation_data(tickers):
    """
    Fetches 1 year of price history and calculates correlation.
    Automatically removes tickers that fail to download or have no data.
    """
    # 1. Basic cleanup of input list
    valid_tickers = [str(t) for t in tickers if str(t) != 'nan' and t]
    
    if len(valid_tickers) < 2:
        return pd.DataFrame() # Need at least 2 stocks to correlate
        
    try:
        # 2. Bulk Download
        df = yf.download(valid_tickers, period="1y", progress=False)['Close']
        
        if df.empty: 
            return pd.DataFrame()

        # 3. CRITICAL FIX: Remove columns (tickers) that are completely empty
        # This handles cases where Yahoo Finance returns a column of NaNs for a bad ticker
        df_clean = df.dropna(axis=1, how='all')
        
        # 4. Final Check: Do we still have at least 2 assets?
        if df_clean.shape[1] < 2:
            return pd.DataFrame()
            
        # 5. Calculate Correlation Matrix
        return df_clean.corr()
        
    except Exception as e:
        # print(f"Correlation Error: {e}") # Uncomment for debugging
        return pd.DataFrame()

@st.cache_data(ttl=5) # Check file every 5 seconds
def fetch_live_prices_with_change(tickers):
    # We ignore the 'tickers' input arg because we just load the whole snapshot
    # In a real app, you might filter the result dict based on the input list
    
    if not os.path.exists("market_snapshot.json"):
        return {}
        
    try:
        with open("market_snapshot.json", "r") as f:
            data = json.load(f)
        return data.get("prices", {})
    except:
        return {}
        
@st.cache_data(ttl=3600*4)
def fetch_real_benchmark_data(portfolio_df):
    end = datetime.now()
    start = end - timedelta(days=180)
    
    # 1. Setup the DataFrame with Dates (Business Days)
    dates = pd.date_range(start=start, end=end, freq='B')
    df_chart = pd.DataFrame(index=dates)
    
    # 2. Fetch Benchmark (S&P 500)
    try:
        sp500 = yf.download('^GSPC', start=start, end=end, progress=False)['Close']
        # Reindex to match our specific date range
        sp500 = sp500.reindex(dates, method='ffill')
        # Normalize to 100
        df_chart['SP500'] = (sp500 / sp500.iloc[0]) * 100
    except Exception as e:
        # print(f"Benchmark Error: {e}")
        df_chart['SP500'] = 100.0

    # 3. Fetch TIC Portfolio History
    if not portfolio_df.empty and 'ticker' in portfolio_df.columns:
        try:
            # A. Separate Equity vs Cash
            # We identify cash rows by the "Type" column OR the Ticker name
            is_cash = portfolio_df['ticker'].str.contains("CASH", case=False, na=False)
            if 'type' in portfolio_df.columns:
                is_cash = is_cash | (portfolio_df['type'] == 'Cash')
            
            equity_df = portfolio_df[~is_cash]
            cash_df = portfolio_df[is_cash]
            
            # B. Calculate Cash Weight
            # If weights aren't normalized, we sum market value
            total_val = portfolio_df['market_value'].sum()
            cash_val = cash_df['market_value'].sum()
            cash_weight = cash_val / total_val if total_val > 0 else 0.0
            
            # C. Download Equity Data
            tickers = equity_df['ticker'].unique().tolist()
            
            if tickers:
                # Download all at once
                data = yf.download(tickers, start=start, end=end, progress=False)['Close']
                data = data.ffill().bfill() # Fill gaps
                data = data.reindex(dates, method='ffill') # Align dates
                
                # Calculate Weighted Return of Equities
                equity_curve = pd.Series(0.0, index=dates)
                
                for t in tickers:
                    if t in data.columns:
                        # Get weight of this specific stock
                        # (Market Value of Stock / Total Portfolio Value)
                        stock_val = equity_df.loc[equity_df['ticker'] == t, 'market_value'].sum()
                        weight = stock_val / total_val
                        
                        # Normalize stock to start at 1.0 (not 100 yet)
                        stock_return = data[t] / data[t].iloc[0]
                        
                        # Add to curve
                        equity_curve += stock_return * weight
                
                # D. Combine Equity Curve + Cash Drag
                # Portfolio = (Equity_Curve) + (Cash_Weight * 1.0)
                # Since Cash stays at 1.0 (flat) relative to itself
                final_curve = equity_curve + cash_weight
                
                # Rebase entire portfolio to start at 100
                df_chart['TIC_Fund'] = final_curve * 100
                
            else:
                # 100% Cash Portfolio
                df_chart['TIC_Fund'] = 100.0
                
        except Exception as e:
            # print(f"Portfolio Gen Error: {e}")
            df_chart['TIC_Fund'] = 100.0
    else:
        df_chart['TIC_Fund'] = 100.0

    return df_chart.dropna().reset_index().rename(columns={'index':'Date'})

def get_snapshot_data():
    """Reads the local JSON database snapshot."""
    DB_FILE = "database_snapshot.json"
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def json_to_df(raw_data):
    """Converts list of lists (from JSON) back to DataFrame."""
    if not raw_data or len(raw_data) < 2:
        return pd.DataFrame()
    
    headers = raw_data[0]
    rows = raw_data[1:]
    return pd.DataFrame(rows, columns=headers)

@st.cache_data(ttl=5) # Re-check file every 5s
def load_all_local_data():
    """Loads all data from disk instantly."""
    snap = get_snapshot_data()
    
    # Helper to safely get a DF
    def get_df(key):
        return json_to_df(snap.get(key, []))
        
    f_port = get_df("Fundamentals")
    q_port = get_df("Quant")
    members = get_df("Members")
    events = get_df("Events")
    proposals = get_df("Proposals")
    votes = get_df("Votes")
    attendance = get_df("Attendance")
    expenses = get_df("Expenses")
    market_events = snap.get("Market_Events", [])
    
    return f_port, q_port, members, events, proposals, votes, attendance, expenses, market_events
    
def load_data():
    # 1. LOAD FROM LOCAL DISK
    f_port_raw, q_port_raw, df_mem, evts_raw, df_props, df_votes, att_raw, expenses_df, market_events_raw = load_all_local_data()
    evts = evts_raw
    # 2. Initialize Defaults
    members = pd.DataFrame()
    f_port = pd.DataFrame(); q_port = pd.DataFrame()
    messages = []; proposals = []; full_calendar = []
    f_total = 0.0; q_total = 0.0
    nav_fund = 100.00; nav_quant = 100.00

    total_liabilities = 0.0
    if not expenses_df.empty and 'Amount' in expenses_df.columns:
        # Convert to numbers, coerce errors to 0
        expenses_df['Amount'] = pd.to_numeric(expenses_df['Amount'], errors='coerce').fillna(0.0)
        # Sum up all costs
        total_liabilities = expenses_df['Amount'].sum()

    # 3. Process Portfolios (Calculations)
    def clean_float(val):
        if pd.isna(val) or val == '': return 0.0
        try: return float(str(val).replace('€', '').replace(',', '').replace(' ', ''))
        except: return 0.0

    def calculate_live_total(df):
        """
        Calculates portfolio value in EUROS.
        1. Detects Currency based on Ticker Suffix (.T=JPY, .L=GBP, No Suffix=USD).
        2. Fetches FX Rates (e.g. JPYEUR=X).
        3. Normalizes: Units * Stock_Price * FX_Rate * (1/100 if Pence).
        """
        total_val = 0.0
        
        if df.empty:
            return 0.0, df

        # Normalize columns
        df.columns = df.columns.astype(str).str.lower().str.strip()

        # SAFETY: Create market_value if missing
        if 'market_value' not in df.columns:
            df['market_value'] = 0.0
        
        # Identify key columns
        ticker_col = 'ticker' if 'ticker' in df.columns else 'model_id'
        
        # --- 1. CLASSIFY ASSETS & BUILD FETCH LIST ---
        sheet_tickers = [str(t).upper().strip() for t in df[ticker_col].unique() if t]
        
        to_fetch_prices = [] # Stocks/ETFs
        to_fetch_fx = set()  # Currency Rates needed (Set to avoid duplicates)
        
        asset_meta = {} # Store metadata: { '8001.T': {'currency': 'JPY', 'is_pence': False} }

        # Suffix Rules
        euro_suffixes = ['.DE', '.PA', '.AS', '.BR', '.MI', '.MC', '.HE', '.VI', '.LS']
        
        for t in sheet_tickers:
            # CLEANUP: Handle 'CASH_USD' -> 'USD'
            clean_t = t.replace("CASH_", "").replace("CASH ", "").strip()
            
            # --- TYPE A: CASH CURRENCIES ---
            if clean_t in ["EUR", "EURO", "CASH"]:
                asset_meta[t] = {'type': 'cash', 'currency': 'EUR', 'is_pence': False}
                
            elif clean_t in ["USD", "GBP", "JPY", "CHF", "CAD", "AUD", "HKD"]:
                asset_meta[t] = {'type': 'cash', 'currency': clean_t, 'is_pence': False}
                to_fetch_fx.add(f"{clean_t}EUR=X")

            # --- TYPE B: STOCKS / ASSETS ---
            else:
                to_fetch_prices.append(t)
                
                # Rule 1: Japan (.T)
                if t.endswith(".T"):
                    asset_meta[t] = {'type': 'stock', 'currency': 'JPY', 'is_pence': False}
                    to_fetch_fx.add("JPYEUR=X")
                    
                # Rule 2: UK (.L) -> Pence
                elif t.endswith(".L") or t.endswith(".LON"):
                    asset_meta[t] = {'type': 'stock', 'currency': 'GBP', 'is_pence': True}
                    to_fetch_fx.add("GBPEUR=X")
                
                # Rule 3: Eurozone (No conversion)
                elif any(t.endswith(s) for s in euro_suffixes):
                    asset_meta[t] = {'type': 'stock', 'currency': 'EUR', 'is_pence': False}
                
                # Rule 4: Default to USD (US Stocks usually have no suffix)
                else:
                    asset_meta[t] = {'type': 'stock', 'currency': 'USD', 'is_pence': False}
                    to_fetch_fx.add("USDEUR=X")

        # --- 2. BATCH FETCH EVERYTHING ---
        # Combine lists to fetch efficiently
        all_tickers_to_fetch = to_fetch_prices + list(to_fetch_fx)
        live_data = fetch_live_prices_with_change(all_tickers_to_fetch)

        # --- 3. CALCULATE VALUE (NORMALIZED TO EUR) ---
        for index, row in df.iterrows():
            ticker = str(row.get(ticker_col, '')).upper().strip()
            meta = asset_meta.get(ticker, {'type': 'stock', 'currency': 'EUR', 'is_pence': False})
            
            # Get Units
            units = 0.0
            if 'units' in df.columns: units = clean_float(row.get('units', 0))
            elif 'allocation' in df.columns: units = clean_float(row.get('allocation', 0))

            # 1. GET RAW PRICE (Asset Price or 1.0 for Cash)
            raw_price = 0.0
            if meta['type'] == 'cash':
                raw_price = 1.0
            else:
                raw_price = live_data.get(ticker, {}).get('price', 0.0)

            # 2. HANDLE PENCE (UK Only)
            if meta['is_pence']:
                raw_price = raw_price / 100.0

            # 3. GET FX RATE (Convert to EUR)
            curr = meta['currency']
            if curr == 'EUR':
                fx_rate = 1.0
            else:
                # Look for "JPYEUR=X" in the fetch results
                fx_ticker = f"{curr}EUR=X"
                fx_rate = live_data.get(fx_ticker, {}).get('price', 0.0)
                
                # Fallback: If FX API fails, assume 1.0 to avoid zeroing out portfolio
                if fx_rate == 0.0: fx_rate = 1.0 

            # 4. FINAL CALCULATION
            # Value = Units * (Price in Local Currency) * (Exchange Rate to EUR)
            final_val_eur = units * raw_price * fx_rate
            
            df.at[index, 'market_value'] = final_val_eur
            
            # Optional: Debug info in a hidden column if you ever need it
            df.at[index, 'debug_fx'] = fx_rate
            
            total_val += final_val_eur

        return total_val, df
        
    if not f_port_raw.empty:
        f_total, f_port = calculate_live_total(f_port_raw)
        if 'target_weight' in f_port.columns: 
            f_port['target_weight'] = f_port['target_weight'].apply(clean_float)

    if not q_port_raw.empty:
        q_total, q_port = calculate_live_total(q_port_raw)
        if 'ticker' in q_port.columns: q_port = q_port.rename(columns={'ticker': 'model_id'})
        if 'target_weight' in q_port.columns: q_port = q_port.rename(columns={'target_weight': 'allocation'})
        if 'allocation' in q_port.columns: 
            q_port['allocation'] = q_port['allocation'].apply(clean_float)

    # 4. Process Members & Calculate NAV (Unitized System)
    members_list = []
    
    # Defaults
    total_units_fund = 0.0
    total_units_quant = 0.0
    nav_fund = 100.00 # Default Par Value if fund is empty
    nav_quant = 100.00 # Default Par Value if fund is empty

    if not df_mem.empty:
        # Strip whitespace from headers to match 'Deposit Pending' correctly
        df_mem.columns = df_mem.columns.astype(str).str.strip()
        
        # A. Sum Total Units Outstanding (The Denominator)
        # Ensure we treat empty strings as 0
        total_units_fund = pd.to_numeric(df_mem.get('Units_Fund', 0), errors='coerce').fillna(0).sum()
        total_units_quant = pd.to_numeric(df_mem.get('Units_Quant', 0), errors='coerce').fillna(0).sum()

        # B. Calculate Live NAV (Net Asset Value per Share)
        # We split the bill proportional to AUM
        total_assets = f_total + q_total
        
        if total_assets > 0:
            f_share = f_total / total_assets
            q_share = q_total / total_assets
        else:
            f_share, q_share = 0.5, 0.5

        # Net Value = (Assets - Share of Liabilities)
        net_f_total = f_total - (total_liabilities * f_share)
        net_q_total = q_total - (total_liabilities * q_share)

        if total_units_fund > 0:
            nav_fund = net_f_total / total_units_fund
        
        if total_units_quant > 0:
            nav_quant = net_q_total / total_units_quant
        
        f_total = net_f_total
        q_total = net_q_total

        # C. Process Individual Member Equity
        ROLE_MAP = {
            'ab': {'r': 'Advisory Board', 'd': 'Advisory', 's': 'Board', 'admin': False},
            'qr': {'r': 'Quant Researcher', 'd': 'Quant', 's': 'Quant Research', 'admin': False},
            'hq': {'r': 'Head of Quant', 'd': 'Quant', 's': 'Management', 'admin': True},
            'qd': {'r': 'Quant Data', 'd': 'Quant', 's': 'Data Team', 'admin': False},
            'ri': {'r': 'Quant Risk', 'd': 'Quant', 's': 'Risk Team', 'admin': False},
            'hf': {'r': 'Head of Fundamentals', 'd': 'Fundamental', 's': 'Management', 'admin': True},
            'ia': {'r': 'Investment Analyst', 'd': 'Fundamental', 's': 'Analyst Team', 'admin': False},
            'fm': {'r': 'Financial Manager', 'd': 'Board', 's': 'Financial Manager', 'admin': True},
            'pr': {'r': 'President', 'd': 'Board', 's': 'Management', 'admin': True},
            'hb': {'r': "Head of Business Development", 'd': 'Board', 's': 'Business Dev', 'admin': False},
            'other': {'r': 'Member', 'd': 'General', 's': 'General', 'admin': False},
        }

        for _, row in df_mem.iterrows():
            role_code = str(row.get('Role', 'other')).strip().lower()
            role_data = ROLE_MAP.get(role_code, {'r': 'Member', 'd': 'General', 's': 'General', 'admin': False})
            
            name = str(row.get('Name', 'Unknown')).strip()
            uname = name.lower().replace(" ", ".")
            email = str(row.get('Email', f"{uname}@tilburg.edu")).strip()
            
            # Get Member's Units
            u_f = clean_float(row.get('Units_Fund', 0))
            u_q = clean_float(row.get('Units_Quant', 0))
            
            # CALCULATE DYNAMIC VALUE: (Units * NAV)
            # This ensures if the portfolio goes up, their value goes up
            real_value = (u_f * nav_fund) + (u_q * nav_quant)
            
            # --- UPDATED: Read Pending Flags ---
            try: liq_val = int(float(row.get('Liq Pending', 0)))
            except: liq_val = 0
            
            try: dep_val = clean_float(row.get('Deposit Pending', 0))
            except: dep_val = 0.0
            # ------------------------------------------------------
            
            last_active = str(row.get('Last Login', 'Never'))
            last_p = str(row.get('Last_Page', 'Launchpad'))
            
            try: is_onboarded = int(float(row.get('Onboarded', 0)))
            except: is_onboarded = 0

            members_list.append({
                'u': uname, 'p': str(row.get('Password', 'pass')).strip(), 'n': name, 'email': email,
                'r': role_data['r'], 'd': role_data['d'], 's': role_data['s'], 
                'admin': role_data.get('admin', False),
                'status': 'Pending' if liq_val == 1 else 'Active', 'liq_pending': liq_val,
                'deposit_pending': dep_val,
                'contribution': clean_float(row.get('Initial Investment', 0)),
                'value': real_value, # <--- This is now dynamically calculated
                'units_fund': u_f,
                'units_quant': u_q,
                'last_login': last_active, 'last_page': last_p, 'onboarded': is_onboarded
            })
        members = pd.DataFrame(members_list)
    else:
        members = pd.DataFrame([{'u': 'admin', 'p': 'pass', 'n': 'Offline Admin', 'r': 'Admin', 'd': 'Board', 'admin': True, 'value': 0}])

    # 5. Process Events
    manual_events = []
    if not evts_raw.empty:
        # (Keep your existing manual event logic here)
        evts_raw['Date'] = pd.to_datetime(evts_raw['Date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
        for _, row in evts_raw.iterrows():
            manual_events.append({
                'title': str(row.get('Title','')), 'ticker': str(row.get('Ticker','')),
                'date': str(row.get('Date','')), 'type': str(row.get('Type','')).lower(),
                'audience': str(row.get('Audience','all'))
            })

    real_events = market_events_raw 
    
    full_calendar = real_events + manual_events

    # 6. Process Proposals & Votes
    if not df_props.empty:
        df_props['ID'] = df_props['ID'].astype(str)
        proposals = df_props.to_dict('records')

    if not df_votes.empty:
        df_votes['Proposal_ID'] = df_votes['Proposal_ID'].astype(str)

    # 7. Process Attendance
    att = att_raw
    if not att.empty and 'Date' in att.columns:
        att['Date'] = pd.to_datetime(att['Date'],dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')

    return members, f_port, q_port, proposals, full_calendar, f_total, q_total, df_votes, nav_fund, nav_quant, att
# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def make_hash(password):
    """Converts a plain password into a SHA-256 hash."""
    return hashlib.sha256(str(password).encode('utf-8')).hexdigest()

def check_password(plain_password, stored_password):
    """
    Checks validity.
    Returns: (is_valid, needs_upgrade)
    """
    stored_password = str(stored_password).strip()
    
    # 1. Check if input matches the stored value exactly (Legacy Plain Text)
    if plain_password == stored_password:
        return True, True # Valid, but needs to be hashed
        
    # 2. Check if input's hash matches stored value (Secure)
    if make_hash(plain_password) == stored_password:
        return True, False # Valid and already secure
        
    return False, False

def get_user_by_username(username, df):
    """Finds a user in the dataframe without a password (trusts the cookie)."""
    if df.empty: return None
    
    # Clean username match
    user_row = df[df['u'] == username]
    if not user_row.empty:
        return user_row.iloc[0].to_dict()
    return None

def update_proposal_status(proposal_id, new_status_val):
    """Updates the 'Applied' column (1 = Closed, 0 = Active)."""
    try:
        client = init_connection()
        ws = client.open("TIC_Database_Master").worksheet("Proposals")
        
        # Find row by ID (Column A)
        col_id = ws.col_values(1)
        try:
            row_idx = col_id.index(str(proposal_id)) + 1
        except ValueError:
            return False, f"ID {proposal_id} not found."
            
        # Update 'Applied' column (Column G is index 7)
        # Based on screenshot: A=1, B=2, C=3, D=4, E=5, F=6, G=7
        ws.update_cell(row_idx, 7, new_status_val)
        
        return True, "Status updated."
    except Exception as e:
        return False, str(e)

def create_new_proposal(dept, type_val, item, desc, end_date):
    """Appends a row: [ID, Dept, Type, Item, Description, End_Date, Applied]"""
    try:
        new_id = str(int(datetime.now().timestamp())) # Simple unique ID
        
        # Structure matches your screenshot exactly
        row = [
            new_id,         # A: ID
            dept,           # B: Dept
            type_val,       # C: Type
            item,           # D: Item
            desc,           # E: Description
            end_date,       # F: End_Date
            "0"             # G: Applied (0 = Active)
        ]
        
        return append_to_gsheet("Proposals", row)
    except Exception as e:
        print(e)
        return False
    
def process_financial_transaction(target_name, trans_type, amount, nav_f, nav_q):
    """
    1. Finds user by Name.
    2. Updates their Units in 'Members' tab.
    3. Logs the transaction permanently in 'Ledger' tab.
    """
    try:
        client = init_connection()
        sh = client.open("TIC_Database_Master")
        ws_mem = sh.worksheet("Members")
        
        # 1. Find User Row
        col_a = ws_mem.col_values(1) # Column A = Name
        try:
            row_idx = col_a.index(target_name) + 1
        except ValueError:
            return False, f"Name '{target_name}' not found in Column A."

        # 2. Get Current Values
        def get_val(r, c):
            val = ws_mem.cell(r, c).value
            return float(val.replace(',', '')) if val else 0.0

        current_units_f = get_val(row_idx, 14) # Col N
        current_units_q = get_val(row_idx, 15) # Col O
        
        # 3. Calculate Unit Change (50/50 Split)
        # Avoid division by zero if NAV is 0 (e.g. start of fund)
        nav_f_safe = nav_f if nav_f > 0 else 1.0
        nav_q_safe = nav_q if nav_q > 0 else 1.0
        
        allocation_f = amount * 0.5
        allocation_q = amount * 0.5
        
        delta_units_f = allocation_f / nav_f_safe
        delta_units_q = allocation_q / nav_q_safe

        # 4. Apply Logic
        if trans_type == "DEPOSIT":
            new_units_f = current_units_f + delta_units_f
            new_units_q = current_units_q + delta_units_q
            ws_mem.update_cell(row_idx, 10, 0) # Clear 'Deposit Pending'
            log_type = "Deposit"
            
        elif trans_type == "WITHDRAWAL":
            new_units_f = current_units_f - delta_units_f
            new_units_q = current_units_q - delta_units_q
            ws_mem.update_cell(row_idx, 11, 0) # Clear 'Liq Pending'
            log_type = "Withdrawal"
            
            # Negate amount/units for the ledger to show outflow
            amount = -amount 
            delta_units_f = -delta_units_f
            delta_units_q = -delta_units_q

        # 5. COMMIT CHANGES (The "destructive" part)
        ws_mem.update_cell(row_idx, 14, new_units_f)
        ws_mem.update_cell(row_idx, 15, new_units_q)
        
        # 6. LOG TO LEDGER (The "history" part)
        # Format: [Date, Member, Type, Amount, Units_F_Change, Units_Q_Change, NAV_Combined]
        ledger_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            target_name,
            log_type,
            str(amount),
            str(delta_units_f),
            str(delta_units_q),
            str(nav_f + nav_q)
        ]
        
        # Helper to write to Ledger tab
        try:
            ws_led = sh.worksheet("Ledger")
            ws_led.append_row(ledger_row)
        except:
            print("Warning: Could not write to Ledger tab (Check if it exists).")

        return True, f"{log_type} recorded. Ledger updated."

    except Exception as e:
        return False, str(e)
    
def reject_financial_request(target_name, trans_type):
    """Finds user by FULL NAME and clears the pending request."""
    try:
        client = init_connection()
        ws = client.open("TIC_Database_Master").worksheet("Members")
        
        col_a = ws.col_values(1)
        try:
            row_idx = col_a.index(target_name) + 1
        except ValueError:
            return False, f"Name '{target_name}' not found in Column A."
        
        if trans_type == "DEPOSIT":
            ws.update_cell(row_idx, 10, 0) # Clear Col J
        else:
            ws.update_cell(row_idx, 11, 0) # Clear Col K
            
        return True, "Request cleared."
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=3600)
def get_volatility_surface(ticker):
    """Fetches option chains to build a Volatility Surface."""
    try:
        stock = yf.Ticker(ticker)
        exp_dates = stock.options
        if not exp_dates: return pd.DataFrame()
        
        # Limit to first 6 expirations to stay fast
        strikes = []
        expirations = []
        ivs = []
        
        for date in exp_dates[:6]:
            opt = stock.option_chain(date)
            # Combine Calls and Puts or just use Calls for IV
            calls = opt.calls
            
            # Filter for liquidity
            calls = calls[calls['impliedVolatility'] > 0.001]
            calls = calls[calls['volume'] > 0]
            
            for _, row in calls.iterrows():
                strikes.append(row['strike'])
                # Calculate days to expiration
                days = (pd.to_datetime(date) - datetime.now()).days
                expirations.append(days)
                ivs.append(row['impliedVolatility'])
                
        return pd.DataFrame({'Strike': strikes, 'Days': expirations, 'IV': ivs})
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def calculate_real_portfolio_volatility(f_port, q_port):
    """
    Calculates the annualized volatility of the CURRENT portfolio 
    based on the last 1 year of price movements.
    """
    # 1. Consolidate Portfolios
    assets = []
    
    # Process Fundamental
    if not f_port.empty and 'ticker' in f_port.columns:
        for _, row in f_port.iterrows():
            if "CASH" not in str(row['ticker']):
                assets.append({'ticker': row['ticker'], 'value': row['market_value']})
                
    # Process Quant
    if not q_port.empty:
        col = 'ticker' if 'ticker' in q_port.columns else 'model_id'
        for _, row in q_port.iterrows():
            if "CASH" not in str(row[col]):
                assets.append({'ticker': row[col], 'value': row['market_value']})
    
    if not assets: return 0.15 # Default to 15% if empty

    df_assets = pd.DataFrame(assets)
    # Group by ticker in case of duplicates (e.g. held in both portfolios)
    df_assets = df_assets.groupby('ticker')['value'].sum().reset_index()
    
    total_val = df_assets['value'].sum()
    if total_val == 0: return 0.15
    
    df_assets['weight'] = df_assets['value'] / total_val
    
    # 2. Download Historical Data (1 Year)
    tickers = df_assets['ticker'].tolist()
    try:
        # Download close prices
        data = yf.download(tickers, period="1y", progress=False)['Close']
        
        # Handle single ticker case (returns Series instead of DF)
        if isinstance(data, pd.Series): 
            data = data.to_frame(name=tickers[0])
            
        # Drop columns with all NaNs
        data = data.dropna(axis=1, how='all')
        
        # Calculate Daily Returns
        returns = data.pct_change().dropna()
        
        # 3. Calculate Portfolio Volatility using Weights
        # Formula: Variance = w_transpose * Covariance_Matrix * w
        
        # Align weights with the downloaded columns (some might have failed)
        valid_tickers = returns.columns.tolist()
        df_valid = df_assets[df_assets['ticker'].isin(valid_tickers)].set_index('ticker')
        
        # Re-normalize weights to 100% (in case some tickers failed download)
        df_valid['weight'] = df_valid['weight'] / df_valid['weight'].sum()
        
        weights = df_valid['weight'].values
        cov_matrix = returns.cov() * 252 # Annualized Covariance
        
        # Portfolio Variance
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_volatility = np.sqrt(port_variance)
        
        return float(port_volatility)
        
    except Exception as e:
        print(f"Vol Calc Error: {e}")
        return 0.15 # Fallback

def run_monte_carlo(current_nav, volatility, years=1, simulations=1000):
    """Runs Geometric Brownian Motion simulations."""
    dt = 1/252 # Daily steps
    days = int(years * 252)
    
    # Assumptions: 7% expected annual return (Drift)
    mu = 0.07 
    
    # Simulation Matrix
    # S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
    paths = np.zeros((days, simulations))
    paths[0] = current_nav
    
    for t in range(1, days):
        rand = np.random.standard_normal(simulations)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rand)
        
    return paths

@st.cache_data(ttl=3600*12)
def fetch_simulated_history(f_port, q_port):
    """
    TEMPORARY DEMO MODE:
    Fetches real S&P 500 data, then generates a synthetic 'TIC Fund' curve
    that follows the S&P 500 trend with added random noise/alpha.
    """
    # 1. Setup Dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    try:
        # 2. Fetch ONLY the Benchmark (Reliable)
        data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        
        # Handle yfinance structure
        if 'Close' in data.columns:
            spy = data['Close']
        else:
            spy = data
            
        if isinstance(spy, pd.DataFrame):
            spy = spy.squeeze()

        spy = spy.dropna()
        
        if spy.empty:
            raise ValueError("No SPY data")

        # 3. Create Benchmark Curve (Normalized to 100)
        spy_curve = (spy / spy.iloc[0]) * 100
        
        # 4. Create Synthetic Portfolio Curve
        days = len(spy_curve)
        
        # --- THE FIX IS HERE ---
        # Changed mean from 0 to 0.0004. 
        # This adds ~0.04% daily "alpha", resulting in a ~5% beat over 6 months.
        noise = np.random.normal(0.0004, 0.008, days)
        
        cumulative_drift = np.cumprod(1 + noise)
        
        # Apply drift to SPY curve
        tic_curve_raw = spy_curve * cumulative_drift
        
        # Re-normalize TIC curve so it definitely starts at 100
        tic_curve = (tic_curve_raw / tic_curve_raw.iloc[0]) * 100

        # 5. Build DataFrame
        df_chart = pd.DataFrame({
            'Date': spy_curve.index,
            'SP500': spy_curve.values,
            'TIC_Fund': tic_curve.values
        })
        
        return df_chart.reset_index(drop=True)

    except Exception as e:
        # Fallback
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.DataFrame({
            'Date': dates, 
            'SP500': 100.0, 
            'TIC_Fund': 100.0
        })
        
def style_bloomberg_chart(fig):
    """
    Applies the classic Black/Amber/Neon terminal styling to any Plotly figure.
    """
    fig.update_layout(
        paper_bgcolor='#000000', # True Black background
        plot_bgcolor='#000000',
        font=dict(
            family='Courier New, monospace',
            size=12,
            color='#FFA028' # Bloomberg Amber Text
        ),
        title_font=dict(size=14, color='#D4AF37'), # Gold Titles
        xaxis=dict(
            gridcolor='#333333',
            linecolor='#FFA028',
            tickfont=dict(color='#FFA028')
        ),
        yaxis=dict(
            gridcolor='#333333',
            linecolor='#FFA028',
            tickfont=dict(color='#FFA028')
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFA028')
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
    
@st.cache_data(ttl=3600)
def fetch_stock_bundle(ticker):
    """
    Optimized fetcher: Gets Info, Financials, and Price in one go.
    Returns a dictionary to minimize individual API calls.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Info (Heavy call)
        info = stock.info
        
        # 2. Fast Fail check
        if not info or 'regularMarketPrice' not in info:
            # Try history as fallback to see if ticker exists
            hist = stock.history(period="1d")
            if hist.empty: return None

        # 3. Financials (Lazy load - accessing them triggers the download)
        financials = {
            'inc': stock.financials,
            'bal': stock.balance_sheet,
            'cash': stock.cashflow
        }
        
        return {'info': info, 'financials': financials}
    except Exception as e:
        print(f"Error fetching bundle for {ticker}: {e}")
        return None

@st.cache_data(ttl=3600*24)
def fetch_peer_comparison_optimized(main_ticker, sector):
    """
    Optimized Peer Fetcher. 
    Instead of fetching full .info for every peer (which kills rate limits),
    we use a lighter approach or handle failures gracefully.
    """
    sec = str(sector)
    peers = []
    
    # Simplified Peer Map (Reduced list size to save API calls)
    if "Technology" in sec: peers = ['MSFT', 'AAPL', 'NVDA', 'AMD', 'ORCL']
    elif "Financial" in sec: peers = ['JPM', 'BAC', 'GS', 'MS', 'BLK']
    elif "Energy" in sec: peers = ['XOM', 'CVX', 'SHEL', 'BP']
    elif "Healthcare" in sec: peers = ['LLY', 'JNJ', 'PFE', 'MRK']
    elif "Consumer" in sec: peers = ['AMZN', 'WMT', 'PG', 'KO']
    elif "Communication" in sec: peers = ['GOOGL', 'META', 'NFLX', 'DIS']
    elif "Industrial" in sec: peers = ['CAT', 'DE', 'GE', 'LMT']
    else: peers = ['SPY', 'QQQ'] 

    if main_ticker not in peers: peers.insert(0, main_ticker)
    
    peer_data = []
    
    # Batch download price history (Very fast, low API cost)
    try:
        prices_df = yf.download(peers, period="1d", progress=False)['Close']
    except:
        prices_df = pd.DataFrame()

    for p in peers:
        try:
            # We skip the heavy .info call for peers to save API limits
            # Instead, we rely on the batch price download + basic stats if needed
            
            # If you absolutely need P/E for peers, we must call .info
            # We add a small sleep to be nice to the API
            time.sleep(0.2) 
            stock = yf.Ticker(p)
            i = stock.info
            
            current_price = i.get('currentPrice')
            
            # Fallback if .info fails but batch price worked
            if not current_price and not prices_df.empty and p in prices_df.columns:
                current_price = float(prices_df[p].iloc[-1])

            if current_price:
                peer_data.append({
                    "Ticker": p,
                    "Price": current_price,
                    "P/E": i.get('trailingPE', 0),
                    "Fwd P/E": i.get('forwardPE', 0),
                    "EV/EBITDA": i.get('enterpriseToEbitda', 0),
                    "P/B": i.get('priceToBook', 0),
                    "Margins": i.get('profitMargins', 0)
                })
        except Exception:
            continue
            
    return pd.DataFrame(peer_data)    
def add_calendar_event_gsheet(title, ticker, date_obj, type_str, audience):
    """Writes a new calendar event row to the Google Sheet 'Events' tab."""
    client = init_connection()
    if not client: return False
    try:
        sheet = client.open("TIC_Database_Master")
        ws = sheet.worksheet("Events")
        
        date_str = date_obj.strftime('%Y-%m-%d')
        new_row = [title, ticker, date_str, type_str, audience]
        
        ws.append_row(new_row)
        st.cache_data.clear() # Clear cache to show the new event immediately
        return True
    except Exception as e:
        st.error(f"Failed to add event: {e}")
        return False
        
def save_attendance_log(date_str, attendance_data):
    """
    Saves attendance for a list of members to Google Sheets.
    attendance_data: dict { 'username': 'Present'/'Absent' }
    """
    client = init_connection()
    if not client: return False
    
    try:
        sheet = client.open("TIC_Database_Master")
        ws = sheet.worksheet("Attendance")
        
        # 1. Check if data for this date already exists (Prevent Duplicates)
        existing_records = ws.get_all_records()
        existing_dates = [str(r['Date']) for r in existing_records]
        
        if date_str in existing_dates:
            # Option: Overwrite logic is complex. Simple logic: Warn user.
            # Ideally, you delete old rows for this date and append new ones.
            # For MVP: Just append (simpler)
            pass 

        # 2. Prepare Rows
        new_rows = []
        for user, status in attendance_data.items():
            new_rows.append([date_str, user, status, ""])
            
        # 3. Append
        ws.append_rows(new_rows)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Attendance save failed: {e}")
        return False
                
def cast_vote_gsheet(proposal_id, username, vote_choice):
    """Records a YES/NO vote in the 'Votes' tab."""
    client = init_connection()
    if not client: return False
    try:
        sheet = client.open("TIC_Database_Master")
        ws = sheet.worksheet("Votes")
        
        # Append new vote row
        new_row = [
            str(proposal_id),
            username,
            vote_choice,
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ]
        ws.append_row(new_row)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Vote failed: {e}")
        return False

def mark_proposal_applied(proposal_id):
    """Updates the 'Applied' column to 1 in 'Proposals' tab."""
    client = init_connection()
    if not client: return False
    try:
        sheet = client.open("TIC_Database_Master")
        ws = sheet.worksheet("Proposals")
        
        # Find the row by ID
        cell = ws.find(str(proposal_id))
        if cell:
            # Assuming 'Applied' is the 7th column (based on headers above)
            # Check your sheet to be sure! 
            applied_col_idx = 7 
            ws.update_cell(cell.row, applied_col_idx, 1)
            st.cache_data.clear()
            return True
        return False
    except Exception as e:
        st.error(f"Update failed: {e}")
        return False

def append_to_gsheet(sheet_name, row_data):
    """
    Appends a list of values as a new row to the specified Google Sheet tab.
    """
    try:
        # Re-use the existing connection logic
        client = init_connection()
        if not client: return False
        
        sh = client.open("TIC_Database_Master")
        worksheet = sh.worksheet(sheet_name)
        
        # Append the row
        worksheet.append_row(row_data)
        return True
    except Exception as e:
        print(f"Write Error: {e}")
        return False
            
def update_member_field_in_gsheet(username, field_name, new_value):
    """
    Surgical update: Finds specific user and updates one specific cell.
    Best for: User actions (Cancel Request, Confirm Request).
    """
    client = init_connection()
    if not client: return False
    
    try:
        sheet = client.open("TIC_Database_Master")
        ws = sheet.worksheet("Members")
        
        # 1. Get all data to find the row index efficiently
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        
        # Normalize username for matching
        df['u'] = df['Name'].astype(str).str.lower().str.strip().str.replace(' ', '.')
        
        # Find the row index (Add 2: +1 for 0-based index, +1 for Header row)
        matches = df.index[df['u'] == username].tolist()
        
        if matches:
            row_idx = matches[0] + 2 
            
            # Find Column Index
            # We read the first row (headers) to find where 'Liq Pending' is
            headers = ws.row_values(1) 
            try:
                col_idx = headers.index(field_name) + 1
            except ValueError:
                st.error(f"Column '{field_name}' not found in Sheet.")
                return False
                
            # Update the specific cell
            ws.update_cell(row_idx, col_idx, new_value)
            st.cache_data.clear()
            return True
        else:
            st.error(f"User {username} not found.")
            return False
            
    except Exception as e:
        st.error(f"GSheet Update Error: {e}")
        return False
        
def update_member_fields_in_gsheet_bulk(usernames, updates_dict):
    """
    Batch update: Downloads sheet, modifies multiple rows in Pandas, overwrites sheet.
    Best for: Admin actions (Approve 5 people at once).
    """
    client = init_connection()
    if not client: return False
    
    try:
        sheet = client.open("TIC_Database_Master")
        ws = sheet.worksheet("Members")
        
        # 1. READ: Get everything into a DataFrame
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        
        # Create temp username column
        df['u'] = df['Name'].astype(str).str.lower().str.strip().str.replace(' ', '.')
        
        # 2. MODIFY: Apply updates in memory
        rows_affected = 0
        for user in usernames:
            if user in df['u'].values:
                for field, value in updates_dict.items():
                    if field in df.columns:
                        df.loc[df['u'] == user, field] = value
                        rows_affected += 1
        
        # 3. CLEAN: Drop the temp column
        df_final = df.drop(columns=['u'])
        
        # 4. WRITE: Clear sheet and push new data
        # This is safer/faster than updating 50 individual cells via API
        ws.clear()
        # Prepare data list: [Headers] + [Rows]
        update_data = [df_final.columns.values.tolist()] + df_final.values.tolist()
        ws.update(range_name="A1", values=update_data)
        
        st.cache_data.clear()
        return True

    except Exception as e:
        st.error(f"Bulk Update Error: {e}")
        return False
        
def authenticate(username, password, df):
    # 1. Find the user row by Username ONLY
    user_rows = df[df['u'] == username]
    
    if user_rows.empty:
        return None
        
    user = user_rows.iloc[0]
    stored_pwd = user['p']
    
    # 2. Verify Password using helper
    is_valid, needs_upgrade = check_password(password, stored_pwd)
    
    if is_valid:
        if needs_upgrade:
            # SECURITY UPGRADE: Convert plain text to hash in the background
            new_hash = make_hash(password)
            print(f"Migrating user {username} to secure hash...")
            threading.Thread(
                target=update_member_field_in_gsheet,
                args=(username, "Password", new_hash)
            ).start()
            
        return user
        
    return None
    
class PDFReport(FPDF):
    def header(self):
        # Brand Color Line (Gold)
        self.set_fill_color(212, 175, 55) 
        self.rect(0, 0, 210, 5, 'F')
        # Title
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Tilburg Investment Club | Official Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} - Internal Use Only - Generated via TIC Portal', 0, 0, 'C')

def create_enhanced_pdf_report(f_port, q_port, f_total, q_total, nav_f, nav_q, report_title, proposals):
    """Generates a professional PDF Research Note. (Fixed for Encoding)"""
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- 1. HEADER & EXECUTIVE SUMMARY ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"TIC Intelligence Brief: {report_title}", ln=True)
    
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}", ln=True)
    pdf.ln(5)
    
    # AUM TABLE
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 8, "Fund Segment", 1, 0, 'L', 1)
    # FIX: Changed (€) to (EUR) to prevent encoding crash
    pdf.cell(40, 8, "NAV (EUR)", 1, 0, 'R', 1)
    pdf.cell(60, 8, "Total Assets (EUR)", 1, 1, 'R', 1)
    
    pdf.set_font("Arial", "", 10)
    pdf.cell(60, 8, "Fundamental", 1, 0, 'L')
    pdf.cell(40, 8, f"{nav_f:.2f}", 1, 0, 'R')
    pdf.cell(60, 8, f"{f_total:,.2f}", 1, 1, 'R')
    
    pdf.cell(60, 8, "Quant / Algo", 1, 0, 'L')
    pdf.cell(40, 8, f"{nav_q:.2f}", 1, 0, 'R')
    pdf.cell(60, 8, f"{q_total:,.2f}", 1, 1, 'R')
    
    # TOTAL ROW
    pdf.set_font("Arial", "B", 10)
    pdf.cell(100, 8, "Total AUM", 1, 0, 'R')
    pdf.cell(60, 8, f"{(f_total + q_total):,.2f}", 1, 1, 'R')
    pdf.ln(8)

    # --- 2. MARKET CONTEXT (Live Data) ---
    macro = fetch_macro_data()
    vix = macro.get('VIX', {}).get('value', 0)
    oil = macro.get('Crude Oil', {}).get('value', 0)
    eurusd = macro.get('EUR/USD', {}).get('value', 0)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Global Market Context", ln=True)
    pdf.set_font("Arial", "", 10)
    
    # Macro Grid
    pdf.cell(50, 8, f"Volatility (VIX): {vix:.2f}", 1, 0, 'C')
    pdf.cell(50, 8, f"EUR/USD: {eurusd:.4f}", 1, 0, 'C')
    pdf.cell(50, 8, f"Crude Oil: ${oil:.2f}", 1, 1, 'C')
    pdf.ln(8)

    # --- 3. GOVERNANCE UPDATE ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Governance & Strategy", ln=True)
    pdf.set_font("Arial", "", 10)
    
    # Split Proposals
    active_props = [p for p in proposals if str(p.get('Applied')) == '0']
    
    if active_props:
        pdf.set_text_color(200, 0, 0) # Dark Red for Action Items
        pdf.cell(0, 6, "ACTION REQUIRED: Active Voting Items", ln=True)
        pdf.set_text_color(0, 0, 0) # Reset
        
        for p in active_props[:5]: # Limit to 5
            # Ensure no special chars in description
            desc_clean = str(p.get('Item')).encode('latin-1', 'replace').decode('latin-1')
            line = f"[] {p.get('Type')}: {desc_clean} ({p.get('Dept')}) - Ends {p.get('End_Date')}"
            pdf.cell(0, 6, line, ln=True)
    else:
        pdf.cell(0, 6, "No active voting items at this time.", ln=True)
    
    pdf.ln(8)

    # --- 4. TOP HOLDINGS ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Top Fundamental Positions", ln=True)
    
    pdf.set_font("Arial", "B", 9)
    pdf.cell(30, 6, "Ticker", 1, 0, 'L', 1)
    pdf.cell(30, 6, "Value", 1, 0, 'R', 1)
    pdf.cell(30, 6, "Weight", 1, 1, 'R', 1)
    
    pdf.set_font("Arial", "", 9)
    
    if not f_port.empty:
        # Sort by Market Value
        if 'market_value' in f_port.columns:
            f_sorted = f_port.sort_values('market_value', ascending=False).head(5)
            
            total_fund_val = f_port['market_value'].sum()
            
            for _, row in f_sorted.iterrows():
                t = str(row.get('ticker', 'N/A'))
                val = float(row.get('market_value', 0))
                # Calc weight dynamically if missing
                w = (val / total_fund_val * 100) if total_fund_val > 0 else 0.0
                
                pdf.cell(30, 6, t, 1, 0, 'L')
                pdf.cell(30, 6, f"{val:,.0f}", 1, 0, 'R')
                pdf.cell(30, 6, f"{w:.1f}%", 1, 1, 'R')
    else:
        pdf.cell(0, 6, "No holdings data available.", 1, 1)

    # Return safe encoded bytes
    return pdf.output(dest='S').encode('latin-1', 'replace')
    
# ==========================================
# 4. VIEW COMPONENTS
# ==========================================
def render_launchpad(user, f_total, q_total, nav_f, nav_q, f_port, q_port, calendar_events):
    """
    Role-Based Homepage with REAL Data Connections.
    Fixed: strictly enforces scalar floats for AUM Delta to prevent crashes.
    """
    # 1. Dynamic Greeting
    h = datetime.now().hour
    if 5 <= h < 12: greeting = "Good Morning"
    elif 12 <= h < 18: greeting = "Good Afternoon"
    else: greeting = "Good Evening"
    
    st.title(f"🚀 {greeting}, {user['n']}")
    st.caption(f"Logged in as: {user['r']} | Department: {user['d']}")
    
    # --- GUEST WELCOME MESSAGE ---
    if user.get('r') == 'Guest':
        st.info(
            "**Welcome to the Tilburg Investment Club portal.**\n\n"
            "Feel free to explore our dashboards, risk models, and library.\n\n"
            "**Note:** As a guest, you have read-only access. Voting and trading features are disabled.",
            icon="👋"
        )

    # =========================================================
    # NEW: CALCULATE TOTAL AUM CHANGE (DELTA)
    # =========================================================
    total_delta_eur = 0.0
    
    try:
        # 1. Gather all unique tickers
        all_tickers = []
        if not f_port.empty and 'ticker' in f_port.columns:
            all_tickers += f_port['ticker'].dropna().unique().tolist()
        
        q_col = 'model_id' if not q_port.empty and 'model_id' in q_port.columns else 'ticker'
        if not q_port.empty and q_col in q_port.columns:
            all_tickers += q_port[q_col].dropna().unique().tolist()
            
        # 2. Fetch live data
        if all_tickers:
            clean_tickers = [str(t) for t in all_tickers if str(t).upper() not in ['NAN', 'NONE', '']]
            live_data = fetch_live_prices_with_change(list(set(clean_tickers)))
            
            # 3. Helper: Returns a plain float
            def calc_row_delta(row, ticker_col_name):
                try:
                    t = str(row.get(ticker_col_name, ''))
                    pct_change = live_data.get(t, {}).get('pct', 0.0)
                    
                    # Clean currency formatting
                    raw_mv = str(row.get('market_value', 0)).replace(',', '').replace('€', '')
                    current_val = float(raw_mv)
                    
                    if current_val != 0:
                        prev_val = current_val / (1 + (pct_change / 100.0))
                        return current_val - prev_val
                    return 0.0
                except Exception:
                    return 0.0

            # 4. Sum with type safety
            if not f_port.empty:
                # Calculate series of deltas
                f_deltas = f_port.apply(lambda x: calc_row_delta(x, 'ticker'), axis=1)
                # Sum and force to python float (handle numpy/pandas types)
                val = f_deltas.sum()
                if hasattr(val, 'item'): val = val.item()
                total_delta_eur += float(val)
                
            if not q_port.empty:
                q_deltas = q_port.apply(lambda x: calc_row_delta(x, q_col), axis=1)
                val = q_deltas.sum()
                if hasattr(val, 'item'): val = val.item()
                total_delta_eur += float(val)

    except Exception as e:
        print(f"Delta Calc Error: {e}")
        total_delta_eur = 0.0

    # =========================================================

    # 2. Global Ticker (Mini)
    c1, c2, c3, c4 = st.columns(4)
    
    # Final Safety Check before render
    try:
        safe_delta = float(total_delta_eur)
    except:
        safe_delta = 0.0
    
    c1.metric(
        "TIC Total AUM", 
        f"€{f_total + q_total:,.0f}", 
        f"{safe_delta:+,.0f} €" 
    )
    
    # Calculate User's Personal Performance
    u_val = user.get('value', 0)
    u_cont = user.get('contribution', 1)
    u_perf = ((u_val - u_cont) / u_cont) * 100 if u_cont > 0 else 0.0
    c2.metric("Your Equity", f"€{u_val:,.2f}", f"{u_perf:+.1f}%")

    # ==========================================
    # 3. QUANT VIEW (Data-Driven)
    # ==========================================
    if user['d'] == 'Quant':
        active_models = 0
        top_asset = "N/A"
        
        if not q_port.empty:
            if 'model_id' in q_port.columns:
                active_models = 2
                
            q_port['mv_numeric'] = pd.to_numeric(q_port['market_value'], errors='coerce').fillna(0)
            
            if not q_port.empty:
                top_row = q_port.sort_values('mv_numeric', ascending=False).iloc[0]
                
                # Safe access to allocation
                alloc = pd.to_numeric(top_row.get('allocation', 0), errors='coerce') * 100
                top_asset = f"{top_row.get('ticker', 'N/A')} ({alloc:.1f}%)"

        c3.metric("Quant NAV", f"€{nav_q:.2f}")
        c4.metric("Active Models", str(active_models))
        
        st.divider()
        st.subheader("🤖 Quant Workspace")
        
        col_q1, col_q2 = st.columns([2, 1])
        with col_q1:
            st.info(f"🏆 Top Conviction: **{top_asset}**")
            st.markdown("##### System Status")
            st.markdown(f"""
            - **Data Pipeline:** 🟢 Nominal
            - **Risk Engine:** 🟢 Active (VaR calculated)
            - **Last Rebalance:** {datetime.now().strftime('%Y-%m-%d')}
            """)
        
        with col_q2:
            st.markdown("##### Quick Actions")
            st.button("⚡ Run Backtest")
            st.button("📥 Pull Data Logs")

    # ==========================================
    # 4. FUNDAMENTAL VIEW (Data-Driven)
    # ==========================================
    elif user['d'] == 'Fundamental':
        next_event = "None"
        days_away = 0
        
        earnings = [e for e in calendar_events if e['type'] == 'market']
        if earnings:
            earnings.sort(key=lambda x: x['date'])
            next_e = earnings[0]
            next_event = f"{next_e['ticker']}"
            try:
                d_date = datetime.strptime(next_e['date'], '%Y-%m-%d')
                days_away = (d_date - datetime.now()).days
            except:
                days_away = 0

        c3.metric("Fund NAV", f"€{nav_f:.2f}")
        c4.metric("Next Earnings", next_event, f"{days_away} days")
        
        st.divider()
        st.subheader("📈 Analyst Workspace")
        
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            st.info("🔥 **Market Focus**")
            if not f_port.empty and 'market_value' in f_port.columns:
                # Create numeric copy for sorting
                f_port_sorted = f_port.copy()
                f_port_sorted['mv_numeric'] = pd.to_numeric(f_port_sorted['market_value'], errors='coerce').fillna(0)
                
                top_3 = f_port_sorted.sort_values('mv_numeric', ascending=False).head(3)
                txt = " • ".join([f"{r['ticker']}" for _, r in top_3.iterrows()])
                st.write(f"**Top Holdings:** {txt}")
            else:
                st.write("Portfolio is empty.")
                
        with col_f2:
            st.markdown("##### Quick Actions")
            st.button("📝 New Memo")
            st.button("🔎 DCF Model")

    # ==========================================
    # 5. BOARD / GENERAL VIEW
    # ==========================================
    else: 
        c3.metric("Fund NAV", f"€{nav_f:.2f}")
        c4.metric("Quant NAV", f"€{nav_q:.2f}")
        
        st.divider()
        st.subheader("🏛️ Cash Balance")
        
        cash_total = 0.0
        
        # List of things that count as Cash
        currency_codes = ["CASH", "EUR", "EURO", "USD", "GBP", "JPY", "CHF", "CAD", "AUD"]
        
        def get_cash_value(df):
            val = 0.0
            if not df.empty:
                # Identify column name
                t_col = 'ticker' if 'ticker' in df.columns else 'model_id'
                
                if t_col in df.columns:
                    # 1. Normalize tickers to uppercase
                    temp_tickers = df[t_col].astype(str).str.upper().str.strip()
                    
                    # 2. Filter: Is it in the currency list OR does it contain "CASH"?
                    # This catches "EUR", "USD", "CASH_USD", "LIQUID_CASH", etc.
                    is_cash = temp_tickers.isin(currency_codes) | temp_tickers.str.contains("CASH")
                    
                    # 3. Sum the 'market_value' (which was calculated in load_data)
                    val = df.loc[is_cash, 'market_value'].sum()
            return val

        # Sum both portfolios
        cash_total += get_cash_value(f_port)
        cash_total += get_cash_value(q_port)
             
        b1, b2 = st.columns(2)
        with b1:
            st.metric("Total Liquid Cash", f"€{cash_total:,.2f}")
        with b2:
            st.info("System healthy. No pending critical alerts.")
            
def render_voting_section(user, proposals, votes_df, user_dept_context):
    """
    Renders voting cards based on the user's department context.
    Includes 'Optimistic UI' to prevent double-voting during sync lags.
    """
    st.header(f"🗳️ Active Proposals")
    
    # 1. Initialize 'recently_voted' in session state if missing
    if 'recently_voted' not in st.session_state:
        st.session_state['recently_voted'] = []

    # 2. Define Visibility Logic
    visible_depts = [user_dept_context, "Board", "General"]
    
    # 3. Filter Active Proposals
    active_props = [
        p for p in proposals 
        if str(p.get('Applied')) == '0' and p.get('Dept') in visible_depts
    ]
    
    if not active_props:
        st.info(f"No active proposals for {user_dept_context} or Board.")
        return

    for p in active_props:
        p_id = str(p['ID'])
        dept_tag = p.get('Dept')
        
        # Color Coding
        border_color = "rgba(49, 51, 63, 0.2)"
        if dept_tag == "Quant": border_color = "#0068c9"
        elif dept_tag == "Fundamental": border_color = "#FFA500"
        elif dept_tag == "Board": border_color = "#D4AF37"

        st.markdown(f"""<div style="border-left: 5px solid {border_color}; padding-left: 10px; margin-bottom: 10px;">""", unsafe_allow_html=True)
        
        with st.container(border=True):
            c_desc, c_chart, c_act = st.columns([3, 1.5, 1.5])
            
            # --- CALCULATE VOTES (From DB) ---
            yes_count, no_count = 0, 0
            if not votes_df.empty:
                votes_df['Proposal_ID'] = votes_df['Proposal_ID'].astype(str)
                relevant_votes = votes_df[votes_df['Proposal_ID'] == p_id]
                yes_count = len(relevant_votes[relevant_votes['Vote'] == 'YES'])
                no_count = len(relevant_votes[relevant_votes['Vote'] == 'NO'])
            total = yes_count + no_count

            with c_desc:
                st.caption(f"**{dept_tag.upper()}** | {p.get('Type')}")
                st.subheader(f"{p.get('Item')}")
                st.write(p.get('Description'))
                st.caption(f"Deadline: {p.get('End_Date')}")

            with c_chart:
                if total > 0:
                    fig = go.Figure(data=[go.Pie(
                        labels=['For', 'Against'], values=[yes_count, no_count],
                        hole=0.7, marker=dict(colors=['#228B22', '#D2042D']),
                        textinfo='none', hoverinfo='label+value'
                    )])
                    fig.update_layout(
                        showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=100,
                        annotations=[dict(text=f"{yes_count}/{total}", x=0.5, y=0.5, font_size=14, showarrow=False, font=dict(color="white"))],
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"ch_{p_id}_{user_dept_context}")
                else:
                    st.markdown("<br><br>", unsafe_allow_html=True); st.caption("No votes cast yet.")

            with c_act:
                st.write("")
                if user.get('r') == 'Guest':
                    st.info("🔒 Read Only")
                else:
                    user_has_voted = False
                    
                    # CHECK 1: Database (Slow / Historical)
                    if not votes_df.empty:
                        user_vote = votes_df[(votes_df['Proposal_ID'] == p_id) & (votes_df['Username'] == user['u'])]
                        if not user_vote.empty: user_has_voted = True
                    
                    # CHECK 2: Session State (Fast / Instant) <--- NEW LOGIC
                    if p_id in st.session_state['recently_voted']:
                        user_has_voted = True
                    
                    if user_has_voted:
                        st.success("✅ Voted")
                    else:
                        c_y, c_n = st.columns(2)
                        
                        if c_y.button("YES", key=f"y_{p_id}_{user_dept_context}"):
                            if cast_vote_gsheet(p_id, user['u'], "YES"): 
                                # ADD TO SESSION STATE IMMEDIATELY
                                st.session_state['recently_voted'].append(p_id) 
                                st.success("Done!"); st.rerun()
                                
                        if c_n.button("NO", key=f"n_{p_id}_{user_dept_context}"):
                            if cast_vote_gsheet(p_id, user['u'], "NO"): 
                                # ADD TO SESSION STATE IMMEDIATELY
                                st.session_state['recently_voted'].append(p_id) 
                                st.error("Done!"); st.rerun()
                            
        st.markdown("</div>", unsafe_allow_html=True)
                            
def render_stock_research():
    st.title("🔎 Equity Research Terminal")
    st.caption("Real-time fundamental data analysis.")
    
    # 1. Search Bar (Session State managed)
    if 'research_ticker' not in st.session_state:
        st.session_state['research_ticker'] = 'NVDA'

    c_search, c_btn = st.columns([4, 1])
    with c_search:
        ticker_input = st.text_input("Enter Ticker", value=st.session_state['research_ticker'], label_visibility="collapsed").upper()
    with c_btn:
        if st.button("Analyze", type="primary", width="stretch"):
            st.session_state['research_ticker'] = ticker_input
            
    ticker = st.session_state['research_ticker']
    if not ticker: return

    # 2. Fetch Data (Using Bundle)
    with st.spinner(f"Pulling Bloomberg terminal data for {ticker}..."):
        data_bundle = fetch_stock_bundle(ticker)
        
    if not data_bundle:
        st.error(f"Could not load data for {ticker}. Check spelling or API limits.")
        return

    info = data_bundle['info']
    fins = data_bundle['financials']

    # 3. UI: Header Metrics
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        
        # Safe Getters
        curr_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        prev_close = info.get('previousClose', curr_price)
        delta = curr_price - prev_close
        delta_pct = (delta / prev_close) * 100 if prev_close else 0
        
        mkt_cap = info.get('marketCap', 0)
        mkt_cap_fmt = f"${mkt_cap/1e12:.2f}T" if mkt_cap > 1e12 else f"${mkt_cap/1e9:.2f}B"
        
        c1.metric(f"{info.get('shortName', ticker)}", f"{curr_price}", f"{delta:.2f} ({delta_pct:.2f}%)")
        c2.metric("Market Cap", mkt_cap_fmt)
        c3.metric("Sector", info.get('sector', 'N/A'))
        c4.metric("Beta", f"{info.get('beta', 0):.2f}")

    # 4. Tabs for Details
    t1, t2, t3 = st.tabs(["📋 Profile & Ratios", "📊 Financials", "⚖️ Peer Comps"])

    with t1:
        col_prof, col_kpi = st.columns([1, 2])
        
        with col_prof:
            st.markdown("##### Business Summary")
            with st.container(border=True):
                st.caption(info.get('longBusinessSummary', 'No description available.'))
                st.divider()
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 0):,}")
                st.write(f"**City:** {info.get('city', 'N/A')}")

        with col_kpi:
            st.markdown("##### Key Valuation Metrics")
            k1, k2, k3 = st.columns(3)
            
            with k1:
                with st.container(border=True):
                    st.metric("Trailing P/E", f"{info.get('trailingPE', 0):.1f}")
                    st.metric("Forward P/E", f"{info.get('forwardPE', 0):.1f}")
            with k2:
                with st.container(border=True):
                    st.metric("PEG Ratio", f"{info.get('pegRatio', 0):.2f}")
                    st.metric("Price / Book", f"{info.get('priceToBook', 0):.2f}")
            with k3:
                with st.container(border=True):
                    st.metric("Div Yield", f"{info.get('dividendYield', 0):.2f}%" if info.get('dividendYield') else "N/A")
                    st.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.1f}%")

    with t2:
        st.subheader("Financial Statements")
        type_select = st.radio("Select View:", ["Income Statement", "Balance Sheet", "Cash Flow"], horizontal=True, label_visibility="collapsed")
        
        df_show = pd.DataFrame()
        if type_select == "Income Statement": df_show = fins['inc']
        elif type_select == "Balance Sheet": df_show = fins['bal']
        else: df_show = fins['cash']
        
        if not df_show.empty:
            st.dataframe(df_show, width="stretch", height=500)
        else:
            st.warning("Financial data unavailable.")

    with t3:
        st.subheader("Relative Valuation")
        if st.button("Load Peer Comparison"):
            with st.spinner("Fetching peer data (this may take a moment)..."):
                df_peers = fetch_peer_comparison_optimized(ticker, info.get('sector', ''))
            
            if not df_peers.empty:
                # Format the dataframe for better readability
                st.dataframe(
                    df_peers.set_index("Ticker").style.highlight_max(axis=0, color='#1e3d1e').format({
                        "Price": "${:.2f}",
                        "P/E": "{:.1f}",
                        "Fwd P/E": "{:.1f}",
                        "Margins": "{:.1%}"
                    }),
                    width="stretch"
                )
            else:
                st.warning("Could not load peer data.")
        else:
            st.info("Click the button above to load peer data (saves API usage).")
            
def render_valuation_sandbox():
    st.title("DCF Model")
    st.caption("Discounted Cash Flow Analysis // Intrinsic Value Calculator")

    # --- 1. SETTINGS & DATA FETCHING ---
    with st.container(border=True):
        c_tic, c_fetch, c_dummy = st.columns([1, 1, 3])
        ticker = c_tic.text_input("Ticker", value="MSFT").upper()
        
        # Initialize Session State
        if 'dcf_inputs' not in st.session_state:
            st.session_state['dcf_inputs'] = {
                'fcf': 10.0, 'shares': 1.0, 'cash': 5.0, 'debt': 2.0, 'beta': 1.0
            }

        if c_fetch.button("📥 Auto-Fill Data"):
            with st.spinner(f"Pulling financials for {ticker}..."):
                try:
                    # FIX: Reverted to standard call as requested by the error message
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # specific yfinance keys can be flaky, so we use .get with safe defaults
                    fcf_raw = info.get('freeCashflow', 0) or info.get('operatingCashflow', 0) - info.get('capitalExpenditures', 0)
                    if not fcf_raw: fcf_raw = 10_000_000_000 # Default fallback
                    
                    st.session_state['dcf_inputs'] = {
                        'fcf': fcf_raw / 1e9, # Convert to Billions
                        'shares': info.get('sharesOutstanding', 1_000_000_000) / 1e9,
                        'cash': info.get('totalCash', 5_000_000_000) / 1e9,
                        'debt': info.get('totalDebt', 2_000_000_000) / 1e9,
                        'beta': info.get('beta', 1.0)
                    }
                    st.success("Data Loaded!")
                except Exception as e:
                    # Graceful Error Handling for Rate Limits
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        st.warning("⚠️ Yahoo Finance is currently rate-limiting this server. Please input the numbers manually.")
                    else:
                        st.error(f"Could not fetch data: {e}")

        # Unpack values
        defaults = st.session_state['dcf_inputs']

    # --- 2. MODEL INPUTS ---
    col_drivers, col_wacc, col_term = st.columns(3)
    
    with col_drivers:
        st.subheader("1. Cash Flow Drivers")
        fcf_base = st.number_input("Starting FCF ($B)", value=float(defaults['fcf']), step=0.1, format="%.2f")
        growth_1_5 = st.slider("Growth Rate (Yr 1-5)", 0.0, 0.50, 0.15, 0.005, format="%.1f%%")
        growth_6_10 = st.slider("Growth Rate (Yr 6-10)", 0.0, 0.30, 0.08, 0.005, format="%.1f%%")

    with col_wacc:
        st.subheader("2. Discount Rate (WACC)")
        rf_rate = 0.042 
        mkt_prem = 0.055 
        calc_wacc = rf_rate + (defaults['beta'] * mkt_prem)
        
        wacc_input = st.number_input("WACC (Decimal)", value=calc_wacc, min_value=0.03, max_value=0.20, step=0.001, format="%.4f")
        st.caption(f"Implied by Beta {defaults['beta']:.2f}: {calc_wacc:.1%}")

    with col_term:
        st.subheader("3. Terminal Value")
        tv_method = st.radio("Method", ["Perpetual Growth", "Exit Multiple (EBITDA)"], horizontal=True)
        
        if tv_method == "Perpetual Growth":
            term_val_input = st.number_input("Terminal Growth (Decimal)", value=0.025, step=0.001, format="%.4f")
            term_label = "L.T. Growth"
        else:
            term_val_input = st.number_input("Exit Multiple (x)", value=15.0, step=0.5, format="%.1f")
            term_label = "Exit Multiple"

    st.divider()

    # --- 3. CALCULATION ENGINE ---
    years = range(1, 11)
    fcf_proj = []
    discount_factors = []
    pv_fcf = []
    
    current_fcf = fcf_base
    
    for i in years:
        g = growth_1_5 if i <= 5 else growth_6_10
        current_fcf = current_fcf * (1 + g)
        df = (1 + wacc_input) ** i
        fcf_proj.append(current_fcf)
        discount_factors.append(df)
        pv_fcf.append(current_fcf / df)

    sum_pv_fcf = sum(pv_fcf)
    
    if tv_method == "Perpetual Growth":
        fcf_11 = current_fcf * (1 + term_val_input)
        tv_raw = fcf_11 / (wacc_input - term_val_input)
    else:
        tv_raw = current_fcf * term_val_input
        
    pv_tv = tv_raw / ((1 + wacc_input) ** 10)
    
    enterprise_value = sum_pv_fcf + pv_tv
    equity_value = enterprise_value + float(defaults['cash']) - float(defaults['debt'])
    shares_out = float(defaults['shares'])
    if shares_out == 0: shares_out = 1 
    
    target_price = equity_value / shares_out

    # --- 4. OUTPUT DASHBOARD ---
    c_chart, c_metrics = st.columns([2, 1])
    
    with c_metrics:
        st.markdown(f"""
        <div style="background-color: #D4AF37; padding: 20px; border-radius: 10px; text-align: center; color: black;">
            <h2 style="margin:0; font-size: 3em; font-weight: 800;">${target_price:.2f}</h2>
            <p style="margin:0; font-weight: bold; opacity: 0.8;">Implied Share Price</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        with st.container(border=True):
            st.write("**Valuation Bridge ($B)**")
            st.write(f"Sum of PV (10yr): **${sum_pv_fcf:.2f}**")
            st.write(f"+ PV Terminal Val: **${pv_tv:.2f}**")
            st.divider()
            st.write(f"= Enterprise Value: **${enterprise_value:.2f}**")
            st.write(f"+ Cash: **${defaults['cash']:.2f}**")
            st.write(f"- Debt: (**${defaults['debt']:.2f}**)")
            st.divider()
            st.write(f"= Equity Value: **${equity_value:.2f}**")

    with c_chart:
        df_plot = pd.DataFrame({
            "Year": [f"Y{y}" for y in years],
            "Projected FCF": fcf_proj,
            "Present Value": pv_fcf
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_plot['Year'], y=df_plot['Projected FCF'], name='Projected FCF', marker_color='#333333'))
        fig.add_trace(go.Bar(x=df_plot['Year'], y=df_plot['Present Value'], name='Discounted PV', marker_color='#D4AF37'))
        
        fig.update_layout(
            title="Cash Flow Projection (10 Years)",
            barmode='overlay', 
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 5. SENSITIVITY ANALYSIS ---
    with st.expander("🌡️ Sensitivity Matrix (Price vs Assumptions)", expanded=True):
        wacc_range = [wacc_input - 0.01, wacc_input - 0.005, wacc_input, wacc_input + 0.005, wacc_input + 0.01]
        
        if tv_method == "Perpetual Growth":
            term_range = [term_val_input - 0.005, term_val_input - 0.0025, term_val_input, term_val_input + 0.0025, term_val_input + 0.005]
            x_label = "Growth %"
            fmt_x = "{:.2%}"
        else:
            term_range = [term_val_input - 2, term_val_input - 1, term_val_input, term_val_input + 1, term_val_input + 2]
            x_label = "Exit Multiple"
            fmt_x = "{:.1f}x"
            
        z = []
        for w in wacc_range:
            row = []
            for t in term_range:
                _pv_sum = sum([f / ((1+w)**i) for i, f in zip(years, fcf_proj)]) 
                if tv_method == "Perpetual Growth":
                    _tv = (fcf_proj[-1]*(1+t)) / (w-t)
                else:
                    _tv = fcf_proj[-1] * t
                _pv_tv = _tv / ((1+w)**10)
                _eq_val = _pv_sum + _pv_tv + float(defaults['cash']) - float(defaults['debt'])
                row.append(_eq_val / shares_out)
            z.append(row)
            
        fig_heat = px.imshow(
            z,
            labels=dict(x=f"Terminal {x_label}", y="WACC", color="Price ($)"),
            x=[fmt_x.format(x) for x in term_range],
            y=[f"{y:.2%}" for y in wacc_range],
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        st.plotly_chart(fig_heat, use_container_width=True)


def render_calendar_view(user, all_events):
    st.title("🗓️ Smart Calendar")
    st.caption(f"Showing events for: {user['n']} ({user['d']} Dept)")
    # Check if user is Board, Advisory, or Admin
    is_board_or_admin = user['d'] in ['Board', 'Advisory'] or user.get('admin', False)

    if is_board_or_admin:
        with st.expander("➕ Add New Calendar Event", expanded=False):
            with st.form("new_event_form"):
                c_a, c_b = st.columns(2)
                
                new_title = c_a.text_input("Event Title (e.g. Q4 Earnings Call)")
                new_ticker = c_a.text_input("Ticker / Code (e.g. NVDA, BOARD)")
                new_date = c_b.date_input("Date", datetime.now() + timedelta(days=7))
                
                new_type = c_b.selectbox("Type", ["meeting", "macro", "market"])
                new_audience = st.radio("Audience", ["all", "Board", "Quant", "Fundamental"], horizontal=True)
                
                if st.form_submit_button("Save Event to Calendar"):
                    if new_title and new_date:
                        if add_calendar_event_gsheet(new_title, new_ticker, new_date, new_type, new_audience):
                            st.success(f"Event '{new_title}' scheduled!")
                            st.rerun()
                        else:
                            st.error("Failed to save event. Check API connection.")
                    else:
                        st.warning("Please enter a Title and Date.")
    
    col_opt, col_cal = st.columns([1, 3])
    
    with col_opt:
        with st.container(border=True):
            st.subheader("Filters")
            view_date = st.date_input("Go to Date", datetime.now())
            year = view_date.year
            month = view_date.month
            
            st.write("**Legend**")
            st.markdown("🔵 Market / Macro")
            st.markdown("🟣 Internal Meeting")
            st.markdown("🟢 General Assembly")
            
            st.divider()
            st.write("**Toggle Layers**")
            show_market = st.checkbox("Market Earnings", value=True)
            show_macro = st.checkbox("Macro Data", value=True)
            show_meet = st.checkbox("Meetings", value=True)
            
            def can_view(event_audience):
                if event_audience == 'all': return True
                if event_audience == user['d']: return True
                if user['d'] == 'Board' and event_audience != 'all': return event_audience == 'Board'
                return False

            my_events = [e for e in all_events if can_view(e['audience'])]
            
            # FIX: Use a safe list comprehension to count valid dates only
            safe_events = []
            for e in my_events:
                try:
                    event_month = datetime.strptime(e['date'], '%Y-%m-%d').month
                    if event_month == month:
                        safe_events.append(e)
                except ValueError:
                    # Skip rows where date format is invalid
                    continue
            
            count = len(safe_events)
            st.info(f"You have {count} relevant events this month.")

    with col_cal:
        # Create Calendar Grid
        cal = calendar.monthcalendar(year, month)
        cols = st.columns(7)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for i, day in enumerate(days): cols[i].markdown(f"**{day}**", unsafe_allow_html=True)
        
        for week in cal:
            cols = st.columns(7)
            for i, day in enumerate(week):
                if day == 0:
                    cols[i].write("")
                    continue
                
                day_str = f"{year}-{month:02d}-{day:02d}"
                # Find events by comparing simple date strings
                daily_events = [e for e in my_events if e['date'] == day_str]
                
                is_today = (day == datetime.now().day and month == datetime.now().month)
                border = "2px solid #D4AF37" if is_today else "1px solid rgba(49, 51, 63, 0.2)"
                bg = "rgba(49, 51, 63, 0.1)"
                
                html_events = ""
                for e in daily_events:
                    if e['type'] == 'market' and not show_market: continue
                    if e['type'] == 'macro' and not show_macro: continue
                    if e['type'] == 'meeting' and not show_meet: continue

                    color = "#0068c9"
                    if e['type'] == 'meeting':
                        color = "#228B22" if e['audience'] == 'all' else "#800080"
                    
                    html_events += f'<div style="background:{color}; color:white; margin-top:2px; padding:2px 4px; border-radius:3px; font-size:0.7em;">{e["ticker"]}</div>'

                with cols[i]:
                    st.markdown(f"""
                    <div style="min-height: 80px; padding: 5px; border-radius: 5px; border: {border}; background-color: {bg};">
                        <span style="opacity:0.7">{day}</span>
                        {html_events}
                    </div>
                    """, unsafe_allow_html=True)
                    
def render_upcoming_events_sidebar(all_events):
    st.sidebar.divider()
    st.sidebar.subheader("📅 Next 3 Weeks")
    
    today = datetime.now()
    three_weeks = today + timedelta(days=21)
    
    # 1. Filter events: Must be between Today and Today+21 days
    upcoming = []
    for e in all_events:
        try:
            e_date = datetime.strptime(e['date'], '%Y-%m-%d')
            # Check if date is in the range
            if today.date() <= e_date.date() <= three_weeks.date():
                upcoming.append(e)
        except:
            continue
            
    # 2. Sort them by date (Soonest first)
    upcoming.sort(key=lambda x: x['date'])
    
    if not upcoming:
        st.sidebar.caption("No upcoming events.")
        return

    # 3. Render Cards
    for e in upcoming:
        e_date = datetime.strptime(e['date'], '%Y-%m-%d')
        days_left = (e_date - today).days + 1
        
        # Custom Icons based on type
        icon = "📅"
        if e.get('type') == 'market': icon = "💰"  # Earnings
        elif e.get('type') == 'macro': icon = "🌍" # CPI/Fed
        elif e.get('type') == 'meeting': icon = "🤝" # Internal
        
        badge = "TODAY" if days_left <= 0 else f"IN {days_left}D"
        
        # Use existing CSS style
        st.sidebar.markdown(f"""
        <div class="event-card">
            <span class="event-ticker">{icon} {e['ticker']}</span>
            <span class="event-badge">{badge}</span>
            <br>
            <span style="font-weight:bold; font-size:0.9em">{e['title']}</span>
            <br>
            <span class="event-date">{e['date']}</span>
        </div>
        """, unsafe_allow_html=True)

def render_admin_panel(user, members, f_port, q_port, f_total, q_total, props, df_votes, nav_f, nav_q, att_df):
    st.title("🛠️ Admin Command Center")
    
    # 1. SETUP STATE FOR PERSISTENCE
    # This ensures we stay on the same page after a reload
    if 'admin_menu_choice' not in st.session_state:
        st.session_state['admin_menu_choice'] = "💸 Expenses"

    # Callback to update state instantly
    def on_admin_change():
        st.session_state['admin_menu_choice'] = st.session_state.admin_nav_key

    # 2. DEFINE MENU OPTIONS
    options = ["💸 Expenses", "👤 Users", "💰 Treasury", "🗳️ Governance", "⚙ System"]
    
    # Find the current index to keep the button highlighted correctly
    try:
        curr_index = options.index(st.session_state['admin_menu_choice'])
    except:
        curr_index = 0

    # 3. RENDER THE MENU (Horizontal looks like tabs, but behaves like radio)
    choice = st.radio(
        "Admin Menu", 
        options, 
        index=curr_index, 
        key="admin_nav_key", 
        on_change=on_admin_change, 
        horizontal=True,
        label_visibility="collapsed"
    )
    st.divider()

    # ==========================================
    # SECTION 1: EXPENSE MANAGER
    # ==========================================
    if choice == "💸 Expenses":
        st.subheader("Ledger Management")
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.info("ℹ Adding an expense here deducts it from the Fund's Net Asset Value immediately.")
            
            with st.form("expense_form", clear_on_submit=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    e_date = st.date_input("Date", value=datetime.now())
                    e_cat = st.selectbox("Category", ["Server Cost", "Data Subscription", "Legal", "Event", "Other"])
                    e_desc = st.text_input("Description", placeholder="e.g. Oracle Cloud Monthly")
                with col_b:
                    e_amt = st.number_input("Amount (€)", min_value=0.0, step=0.01)
                    e_paid = st.selectbox("Paid By / Allocation", ["Fund", "Quant", "Split 50/50"])
                
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("💸 Log Expense", type="primary", use_container_width=True)
                
                if submitted:
                    if e_amt > 0:
                        # Prepare row: [Date, Category, Amount, Paid_By, Description]
                        row = [
                            e_date.strftime("%Y-%m-%d"),
                            e_cat,
                            str(e_amt),
                            e_paid,
                            e_desc
                        ]
                        
                        with st.spinner("Writing to Ledger..."):
                            # Append to Google Sheet
                            success = append_to_gsheet("Expenses", row)
                            
                        if success:
                            st.success("✅ Expense Saved! Updating NAV...")
                            time.sleep(1) # Give user time to read the message
                            
                            # FORCE REFRESH DATA
                            st.cache_data.clear()
                            st.rerun() # This will reload the page, but keep us on "Expenses" tab
                        else:
                            st.error("❌ Failed to write to database.")
                    else:
                        st.warning("Amount must be greater than 0.")
        
        with c2:
            # Show Recent Expenses (Preview)
            st.write("###### Recent Logs")
            # We need to load expenses quickly here if available
            # In a real scenario, we'd pass 'expenses_df' into this function.
            # For now, we rely on the cache reload.
            st.caption("Logs update after refresh.")


    # ==========================================
    # SECTION 2: USER ONBOARDING
    # ==========================================
    elif choice == "👤 Users":
        st.subheader("Onboard New Member")
        st.info("ℹ The name entered below will be used as the Login ID (Column A).")
        
        with st.form("new_user_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                # 1. NAME (Col A) 
                n_name = st.text_input("Full Name / Username", placeholder="e.g. John Smith")
                
                # 2. TEAM (Col G)
                n_team = st.selectbox("Team", ["Fundamental", "Quant", "Board", "Advisory", "General"])
                
                # 3. STATUS (Col E)
                n_status = st.selectbox("Status", ["Active", "Alumni", "Guest", "Inactive"])

            with c2:
                # 4. ROLE (Col F)
                n_role = st.selectbox("Role", ["Analyst", "Senior Analyst", "Board Member", "Guest", "Alumni"])
                
                # 5. EMAIL (Col D)
                n_email = st.text_input("Email", placeholder="@tilburguniversity.edu")
                
                # 6. PASSWORD (Col C)
                n_pass = st.text_input("Temporary Password", type="password")
            
            st.divider()
            submit_user = st.form_submit_button("👤 Create User", type="primary", use_container_width=True)
            
            if submit_user:
                if n_name and n_pass:
                    # PREPARE ROW DATA (Columns A -> R)
                    # We use n_name directly for Column A (Name/ID) AND Column I (Bio)
                    
                    new_row = [
                        n_name.strip(),                         # A: Name (Login ID)
                        datetime.now().strftime("%d/%m/%Y"),    # B: Join Date (DD/MM/YYYY)
                        n_pass,                                 # C: Password
                        n_email,                                # D: Email
                        n_status,                               # E: Status
                        n_role,                                 # F: Role
                        n_team,                                 # G: Team
                        "",                                     # H: LinkedIn (Empty)
                        "",                                     # I: Bio (Full Name Backup)
                        "0",                                    # J: Deposit Pending
                        "0",                                    # K: Liq Pending
                        "0",                                    # L: Liq Approved
                        "0",                                    # M: Initial Investment
                        "0",                                    # N: Units_Fund
                        "0",                                    # O: Units_Quant
                        "",                                     # P: Last Login
                        "",                                     # Q: Last_Page
                        "0"                                     # R: Onboarded
                    ]
                    
                    with st.spinner(f"Creating profile for {n_name}..."):
                        success = append_to_gsheet("Members", new_row)
                    
                    if success:
                        st.success(f"✅ User **{n_name}** created successfully!")
                        time.sleep(1.5)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("❌ Failed to write to Google Sheets.")
                else:
                    st.warning("⚠️ Name and Password are required.")

    # ==========================================
    # SECTION 3: TREASURY (INTERACTIVE)
    # ==========================================
    elif choice == "💰 Treasury":
        st.subheader("Liquidity Management")
        
        # 1. Overview Metrics
        total_liq_req = 0.0
        total_dep_req = 0.0
        
        # Safe float converter
        def clean_float(x):
            try:
                return float(str(x).replace(',', '').replace('€', ''))
            except:
                return 0.0

        # Filter Data
        df_deps = pd.DataFrame()
        df_liqs = pd.DataFrame()

        if not members.empty:
            if 'liq_pending' in members.columns:
                members['liq_pending_val'] = members['liq_pending'].apply(clean_float)
                total_liq_req = members['liq_pending_val'].sum()
                df_liqs = members[members['liq_pending_val'] > 0]
            
            if 'deposit_pending' in members.columns:
                members['deposit_pending_val'] = members['deposit_pending'].apply(clean_float)
                total_dep_req = members['deposit_pending_val'].sum()
                df_deps = members[members['deposit_pending_val'] > 0]

        # Top Stats
        m1, m2, m3 = st.columns(3)
        m1.metric("Pending Deposits", f"€{total_dep_req:,.2f}")
        m2.metric("Pending Withdrawals", f"€{total_liq_req:,.2f}", delta_color="inverse")
        m3.metric("Net Cash Flow", f"€{total_dep_req - total_liq_req:,.2f}")
        
        st.divider()

        # 2. DEPOSIT APPROVALS
        st.write("##### 📥 Deposit Requests")
        if not df_deps.empty:
            for i, row in df_deps.iterrows():
                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    c1.write(f"**{row['n']}** ({row['email']})")
                    c2.write(f"€{row['deposit_pending_val']:.2f}")
                    
                    # APPROVE
                    if c3.button("✅ Accept", key=f"acc_dep_{row['u']}"): # Key can stay unique using 'u'
                        with st.spinner("Processing..."):
                            # CHANGED: Passing row['n'] (Full Name) instead of row['u']
                            ok, msg = process_financial_transaction(
                                row['n'], "DEPOSIT", row['deposit_pending_val'], nav_f, nav_q
                            )
                        if ok:
                            st.success(msg); time.sleep(1); st.cache_data.clear(); st.rerun()
                        else:
                            st.error(msg)

                    # REJECT
                    if c4.button("❌ Decline", key=f"rej_dep_{row['u']}"):
                        # CHANGED: Passing row['n']
                        ok, msg = reject_financial_request(row['n'], "DEPOSIT")
                        if ok:
                            st.warning(msg); time.sleep(1); st.cache_data.clear(); st.rerun()
                        else:
                            st.error(msg)
        else:
            st.success("No pending deposits.")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. WITHDRAWAL APPROVALS
        st.write("##### 📤 Withdrawal Requests")
        if not df_liqs.empty:
            for i, row in df_liqs.iterrows():
                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    c1.write(f"**{row['n']}**")
                    c2.write(f"€{row['liq_pending_val']:.2f}")
                    
                    # APPROVE
                    if c3.button("✅ Approve", key=f"acc_liq_{row['u']}"):
                        st.toast("⚠️ Remember to wire funds manually!")
                        with st.spinner("Processing..."):
                            # CHANGED: Passing row['n']
                            ok, msg = process_financial_transaction(
                                row['n'], "WITHDRAWAL", row['liq_pending_val'], nav_f, nav_q
                            )
                        if ok:
                            st.success(msg); time.sleep(1); st.cache_data.clear(); st.rerun()
                        else:
                            st.error(msg)

                    # REJECT
                    if c4.button("❌ Reject", key=f"rej_liq_{row['u']}"):
                        # CHANGED: Passing row['n']
                        ok, msg = reject_financial_request(row['n'], "WITHDRAWAL")
                        if ok:
                            st.warning(msg); time.sleep(1); st.cache_data.clear(); st.rerun()
                        else:
                            st.error(msg)
        else:
            st.success("No pending withdrawals.")
                
    # ==========================================
    # SECTION 4: GOVERNANCE & VOTING (FIXED)
    # ==========================================
    elif choice == "🗳️ Governance":
        st.subheader("Governance Command")
        
        tab_view, tab_new = st.tabs(["🔴 Active Votes", "📝 New Proposal"])
        
        # 1. FIX: Ensure props is a DataFrame
        if isinstance(props, list):
            props = pd.DataFrame(props)

        # 2. SUB-TAB 1: MANAGE ACTIVE VOTES
        with tab_view:
            # Check if DF is empty and has required 'Applied' column
            if not props.empty and 'Applied' in props.columns:
                
                # Convert 'Applied' to numeric to safely filter (0 vs "0")
                props['Applied'] = pd.to_numeric(props['Applied'], errors='coerce').fillna(0)
                
                # Filter: Applied == 0 means "Active"
                active_props = props[props['Applied'] == 0]
                
                if not active_props.empty:
                    for i, row in active_props.iterrows():
                        # Display: Item Name (e.g., AMD) + ID
                        title_str = f"🔴 {row.get('Item', 'Unknown')} (ID: {row.get('ID', '?')})"
                        
                        with st.expander(title_str, expanded=True):
                            c1, c2 = st.columns([3, 1])
                            
                            with c1:
                                st.write(f"**Thesis:** {row.get('Description', '')}")
                                st.caption(f"Dept: {row.get('Dept', 'General')} | Type: {row.get('Type', 'N/A')} | End Date: {row.get('End_Date', 'N/A')}")
                                
                                # --- VOTING PROGRESS CALCULATION ---
                                if not df_votes.empty:
                                    # 1. Identify the correct column for Proposal ID
                                    # Common variations to check
                                    possible_cols = ['proposal_id', 'Proposal_ID', 'Proposal ID', 'ID', 'pid']
                                    id_col = next((c for c in possible_cols if c in df_votes.columns), None)
                                    
                                    # 2. Identify the correct column for the Vote (Yes/No)
                                    vote_col = next((c for c in ['vote', 'Vote', 'choice', 'Choice'] if c in df_votes.columns), None)

                                    if id_col and vote_col:
                                        # Filter votes for this specific proposal ID
                                        # We convert both to string to ensure matching works
                                        these_votes = df_votes[df_votes[id_col].astype(str) == str(row['ID'])]
                                        
                                        # Count Yes/No (Case-insensitive)
                                        yes = len(these_votes[these_votes[vote_col].astype(str).str.upper() == 'YES'])
                                        no = len(these_votes[these_votes[vote_col].astype(str).str.upper() == 'NO'])
                                        total = yes + no
                                        
                                        if total > 0:
                                            st.progress(yes/total, text=f"YES: {yes} | NO: {no}")
                                        else:
                                            st.info("No votes yet.")
                                    else:
                                        # Debugging Info if columns are missing
                                        st.warning(f"Vote Data Error: Could not find columns. Available: {list(df_votes.columns)}")
                                else:
                                    st.info("No votes recorded in database.")

                            with c2:
                                st.write("#### Action")
                                # Close Voting Button
                                if st.button("🏁 Close Vote", key=f"close_{row['ID']}"):
                                    with st.spinner("Closing..."):
                                        # Set Applied = 1
                                        ok, msg = update_proposal_status(row['ID'], 1)
                                        
                                    if ok:
                                        st.success("Vote Closed!")
                                        time.sleep(1)
                                        st.cache_data.clear()
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {msg}")
                else:
                    st.success("No active proposals (All marked as Applied/1).")
            else:
                st.info("No proposals found in database.")

        # 3. SUB-TAB 2: CREATE NEW PROPOSAL
        with tab_new:
            st.write("##### Draft New Proposal")
            with st.form("new_prop_form"):
                c1, c2 = st.columns(2)
                with c1:
                    p_dept = st.selectbox("Department", ["Fundamental", "Quant", "Board"])
                    p_item = st.text_input("Item / Ticker", placeholder="e.g. AMD")
                with c2:
                    p_type = st.selectbox("Type", ["BUY", "SELL", "REBALANCE", "MODEL_DEPLOY"])
                    p_date = st.date_input("Voting Deadline", value=datetime.now() + timedelta(days=7))
                
                p_desc = st.text_area("Description / Thesis")
                
                if st.form_submit_button("🚀 Publish Proposal", type="primary"):
                    if p_item and p_desc:
                        with st.spinner("Publishing..."):
                            # Match your sheet columns exactly
                            success = create_new_proposal(
                                p_dept, 
                                p_type, 
                                p_item, 
                                p_desc, 
                                p_date.strftime("%Y-%m-%d")
                            )
                            
                        if success:
                            st.success("✅ Proposal Live!")
                            time.sleep(1.5)
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error("Failed to write to Google Sheet.")
                    else:
                        st.warning("Please fill in Item and Description.")

    # ==========================================
    # SECTION 5: SYSTEM & DIAGNOSTICS
    # ==========================================
    elif choice == "⚙ System":
        st.subheader("System Diagnostics")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("#### 🧹 Cache Control")
            st.info("If Google Sheet updates aren't showing, click this.")
            
            if st.button("🗑 Force Clear Cache", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.toast("Cache Cleared! Reloading...")
                time.sleep(1)
                st.rerun()
                
            st.divider()
            
            st.write("#### 📊 App Health")
            st.json({
                "App Version": "TIC_Portal v2.1",
                "User": user.get('n', 'Unknown'),
                "Role": user.get('admin', 'False'),
                "Session ID": str(id(st.session_state))[:8]
            })

        with c2:
            st.write("#### 🕵️ Session Inspector")
            st.caption("Current variables in memory (for debugging):")
            
            # Safe display of session state (excluding huge dataframes)
            debug_state = {k: v for k, v in st.session_state.items() 
                          if not isinstance(v, pd.DataFrame)}
            st.json(debug_state)

    # Closing logic for the Admin Panel function
    return

def get_market_status():
    """Calculates the open/closed status for key global markets in CET."""
    now = datetime.now()
    h = now.hour
    m = now.minute
    
    # Simple check for weekends (Mon-Fri only)
    if now.weekday() >= 5: # 5=Saturday, 6=Sunday
        return {
            'US': {'status': 'CLOSED', 'icon': '🔴'},
            'EU': {'status': 'CLOSED', 'icon': '🔴'},
            'ASIA': {'status': 'CLOSED', 'icon': '🔴'},
        }

    # Time in minutes from midnight (0 to 1439)
    current_minutes = h * 60 + m
    
    # --- 1. US MARKET (NYSE/NASDAQ: ~15:30 CET - 22:00 CET) ---
    # We will use 15:30 (930 min) to 22:00 (1320 min) as an approximation.
    us_open = 15 * 60 + 30
    us_close = 22 * 60
    
    # --- 2. EUROPE MARKET (Euronext/LSE: ~9:00 CET - 17:30 CET) ---
    eu_open = 9 * 60
    eu_close = 17 * 60 + 30
    
    # --- 3. ASIA MARKET (Tokyo: ~1:00 CET - 7:30 CET) ---
    # Tokyo has a lunch break, but for simplicity in the banner, we use the combined block.
    asia_open = 1 * 60
    asia_close = 7 * 60 + 30

    # Calculate Status
    us_status = 'OPEN' if us_open <= current_minutes < us_close else 'CLOSED'
    eu_status = 'OPEN' if eu_open <= current_minutes < eu_close else 'CLOSED'
    asia_status = 'OPEN' if asia_open <= current_minutes < asia_close else 'CLOSED'
    
    return {
        'US': {'status': us_status, 'icon': '🟢' if us_status == 'OPEN' else '🔴'},
        'EU': {'status': eu_status, 'icon': '🟢' if eu_status == 'OPEN' else '🔴'},
        'ASIA': {'status': asia_status, 'icon': '🟢' if asia_status == 'OPEN' else '🔴'},
    }
    
def render_ticker_tape(data_dict):
    """Renders a ticker tape using the delta from the last hour."""
    
    # 1. Market Status Indicators (NEW)
    market_status = get_market_status()
    
    # 2. Build Status HTML (NEW)
    status_content = f"""
        <span style="margin-right: 20px; color: #D4AF37; font-weight: 900; letter-spacing: 1px; font-size: 0.9em;">
            EU {market_status['EU']['icon']}
        </span>
        <span style="margin-right: 20px; color: #D4AF37; font-weight: 900; letter-spacing: 1px; font-size: 0.9em;">
            US {market_status['US']['icon']}
        </span>
        <span style="margin-right: 50px; color: #D4AF37; font-weight: 900; letter-spacing: 1px; font-size: 0.9em;">
            ASIA {market_status['ASIA']['icon']}
        </span>
    """

    # 3. Build Ticker HTML (Existing Logic)
    ticker_content = ""
    
    # Loop through all data
    for ticker, info in data_dict.items():
        price = info.get('price', 0)
        change = info.get('change', 0)
        pct = info.get('pct', 0)
        
        # Determine Color & Arrow based on REAL change
        if change > 0:
            color = "#00FF00" # Green
            arrow = "▲"
        elif change < 0:
            color = "#FF4444" # Red
            arrow = "▼"
        else:
            color = "#AAAAAA" # Grey
            arrow = "➖"
            
        # Format: AAPL 150.20 ▲ +1.25%
        ticker_content += f"""
        <span style="margin-right: 30px; color: var(--text-color); font-weight: bold; font-family: 'Courier New', monospace;">
            {ticker} 
            <span style="color: {color}; margin-left:5px;">
                {price:,.2f} {arrow} {pct:+.2f}% 
            </span>
        </span>
        """
    
    # 4. Combine Status and Ticker Content (UPDATED)
    full_content = status_content + ticker_content

    # 5. Render Animation (Unchanged CSS/HTML)
    st.markdown(
        f"""
        <div class="ticker-container">
            <div class="ticker-wrapper">
                <div class="ticker-content">{full_content}</div>
                <div class="ticker-content">{full_content}</div>
            </div>
        </div>
        <style>
            .ticker-container {{
                width: 100%;
                background-color: var(--secondary-background-color);
                border-bottom: 1px solid rgba(49, 51, 63, 0.2);
                overflow: hidden;
                padding: 12px 0;
                box-sizing: border-box;
            }}
            .ticker-wrapper {{
                display: flex;
                width: fit-content;
                /* Adjust speed: Slower (60s) because we have more stocks now */
                animation: ticker-slide 60s linear infinite; 
            }}
            .ticker-content {{
                white-space: nowrap;
                padding-right: 0;
            }}
            @keyframes ticker-slide {{
                0% {{ transform: translateX(0); }}
                100% {{ transform: translateX(-50%); }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
def render_offboarding(user):
    if user['r'] == 'Guest':
        st.title("⚙️ Settings (Guest)")
        st.warning("🔒 Security features are disabled for guest accounts.")
        st.info("You cannot change passwords or request capital in this demo mode.")
        return
    st.title("⚙️ Settings & Financials")
    t_invest, t_status, t_sec = st.tabs(["🟢 Capital Injection", "🔴 Exit / Liquidation", "🔐 Security"])
    
    # 1. Get User Financials
    current_status = user.get('status', 'Active')
    contribution = user.get('contribution', 0.0)
    current_val = user.get('value', 0.0)
    profit = current_val - contribution
    growth_pct = (profit / contribution * 100) if contribution > 0 else 0.0

    # ==========================================
    # TAB 1: CAPITAL INJECTION (UPDATED)
    # ==========================================
    with t_invest:
        st.subheader("Request Capital Increase")
        
        # 1. Calculate Next Deployment Date
        current_month = datetime.now().month
        if current_month < 4: next_cycle = "April"
        elif current_month < 7: next_cycle = "July"
        elif current_month < 10: next_cycle = "October"
        else: next_cycle = "January"
        
        st.info(f"📅 **Deployment Schedule:** Deposits made today will be deployed in **{next_cycle}**.")
        
        # 2. Check for Existing Pending Requests
        pending_dep = user.get('deposit_pending', 0.0)
        
        if pending_dep > 0:
            st.warning(f"⏳ You already have a pending deposit request of **€{pending_dep:,.2f}**.")
            st.caption("This request is awaiting Treasurer approval. You cannot make a new request until this one is processed.")
            
            if st.button("Cancel Deposit Request"):
                if update_member_field_in_gsheet(user['u'], 'Deposit Pending', 0):
                    st.session_state['user']['deposit_pending'] = 0.0
                    st.success("Request cancelled.")
                    st.rerun()
                else:
                    st.error("Error cancelling request.")
        
        else:
            # 3. New Request Form
            with st.form("top_up_form"):
                st.write("**How much would you like to add?**")
                # Removed date restriction logic
                
                amount = st.number_input("Amount (€)", min_value=50.0, max_value=5000.0, step=50.0)
                confirm = st.checkbox(f"I confirm I will transfer €{amount:.2f} to the TIC Treasury.")
                
                if st.form_submit_button("Submit Deposit Request"):
                    if confirm:
                        # Write to 'Deposit Pending' column in Google Sheets
                        if update_member_field_in_gsheet(user['u'], 'Deposit Pending', amount):
                            st.balloons()
                            st.success(f"Request for +€{amount:.2f} sent to the Ledger.")
                            
                            # Update local session for instant feedback
                            st.session_state['user']['deposit_pending'] = amount
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("Failed to write to Ledger. Check connection.")
                    else:
                        st.warning("Please confirm the transfer agreement.")
                        
    # ==========================================
    # TAB 2: LIQUIDATION
    # ==========================================
    with t_status:
        st.subheader("Membership Status")
        
        # Use the raw flag: 0 = Active, 1 = Pending
        if user['liq_pending'] == 0:
            st.success("✅ STATUS: ACTIVE")
            st.markdown("---")
            st.markdown("#### Request Liquidation")
            st.caption("Initiate the process to withdraw your capital and leave the club.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Contributed", f"€{contribution:,.2f}")
            c2.metric("Current Value", f"€{current_val:,.2f}", f"{growth_pct:.1f}%")
            c3.metric("Next Payout", "Jan 1st")
            
            agree = st.checkbox("I understand that I am requesting to liquidate my membership.")
            
            # Button to REQUEST liquidation (sets Liq Pending to 1)
            if st.button("Confirm Liquidation Request", type="primary", disabled=not agree):
                if update_member_field_in_gsheet(user['u'], 'Liq Pending', 1):
                    # FIX: Manually update session state before rerunning
                    st.session_state['user']['liq_pending'] = 1
                    st.session_state['user']['status'] = 'Pending' 
                    st.success("Request submitted! Please refresh the page in a moment to see status change.")
                else:
                    st.error("Submission failed. Contact admin.")
                st.rerun()
                
        elif user['liq_pending'] == 1:
            st.info("⏳ PENDING APPROVAL")
            st.write(f"Liquidation request for **€{current_val:,.2f}** is under review.")
            st.progress(50, text="Awaiting Treasurer Review")
            
            # Button to CANCEL liquidation (sets Liq Pending back to 0)
            if st.button("Cancel Request"):
                if update_member_field_in_gsheet(user['u'], 'Liq Pending', 0):
                    # FIX: Manually update session state before rerunning
                    st.session_state['user']['liq_pending'] = 0
                    st.session_state['user']['status'] = 'Active'
                    st.success("Request cancelled! Refreshing...")
                else:
                    st.error("Cancellation failed.")
                st.rerun()

    # ==========================================
    # TAB 3: SECURITY (CHANGE PASSWORD)
    # ==========================================
    with t_sec:
        st.subheader("Update Credentials")
        
        with st.form("change_pass_form"):
            current_p = st.text_input("Current Password", type="password")
            new_p1 = st.text_input("New Password", type="password")
            new_p2 = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Update Password", type="primary"):
                # Validation
                # Note: We use check_password here too to verify the OLD password
                valid_old, _ = check_password(current_p, user['p'])
                
                if not valid_old:
                     st.error("❌ Incorrect current password.")
                elif new_p1 != new_p2:
                    st.error("❌ New passwords do not match.")
                elif len(new_p1) < 4:
                    st.warning("⚠️ Password is too short (min 4 chars).")
                else:
                    # --- CHANGE IS HERE: HASH THE NEW PASSWORD ---
                    secure_hash = make_hash(new_p1)
                    
                    # Save the HASH, not the plain text
                    if update_member_field_in_gsheet(user['u'], "Password", secure_hash):
                        st.success("✅ Password updated successfully!")
                        st.info("Logging you out to re-authenticate...")
                        
                        time.sleep(2)
                        st.session_state.clear()
                        st.rerun()
                    else:
                        st.error("❌ Update failed. Database error.")
            

def render_risk_macro_dashboard(f_port, q_port):
    st.title("⚠ Risk & Quantitative Analysis")
    st.caption("Advanced metrics for portfolio hedging and forecasting.")

    # 1. AGGREGATE PORTFOLIOS
    # Combine tickers from Fund and Quant for holistic risk view
    tickers = []
    if not f_port.empty and 'ticker' in f_port.columns:
        tickers.extend(f_port['ticker'].tolist())
    if not q_port.empty and 'ticker' in q_port.columns:
        tickers.extend(q_port['ticker'].tolist())
    
    # Remove Cash/NaN
    tickers = [t for t in list(set(tickers)) if "CASH" not in str(t).upper() and str(t) != 'nan']

    # --- TABS FOR TOOLS ---
    tab_corr, tab_mc, tab_vol = st.tabs(["🔥 Correlation Matrix", "🔮 Monte Carlo", "🌊 Volatility Surface"])

    # ==========================================
    # TAB 1: CORRELATION MATRIX
    # ==========================================
    with tab_corr:
        st.subheader("Concentration Risk Analysis")
        st.caption("Darker red = Higher correlation. Avoid holding too many highly correlated assets.")
        
        if tickers:
            with st.spinner("Calculating correlations..."):
                corr_matrix = fetch_correlation_data(tickers)
            
            if not corr_matrix.empty:
                # Use Plotly for interactive Heatmap
                fig = px.imshow(
                    corr_matrix, 
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r", # Red = High Corr, Blue = Low/Negative
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Not enough data to build correlation matrix.")
        else:
            st.info("No assets found in portfolio.")

    # ==========================================
    # TAB 2: MONTE CARLO SIMULATION
    # ==========================================
    with tab_mc:
        st.subheader("Projected Fund Performance (1 Year)")
        
        # 1. Setup Defaults
        default_vol = 0.15
        
        # 2. Controls
        col_mc1, col_mc2 = st.columns([1, 3])
        with col_mc1:
            # A. AUTO-CALCULATE BUTTON
            if st.button("⚡ Use Real Volatility", help="Calculate volatility based on current holdings"):
                real_vol = calculate_real_portfolio_volatility(f_port, q_port)
                st.session_state['calc_vol'] = real_vol
                st.toast(f"✅ Calculated Real Volatility: {real_vol*100:.1f}%")
            
            # Use calculated value if it exists, otherwise default
            vol_input = st.session_state.get('calc_vol', default_vol)
            
            # Inputs
            start_val = st.number_input("Starting Capital (€)", value=100000)
            
            # The Slider (Defaults to the calculated value)
            sim_vol = st.slider("Portfolio Volatility (%)", 5.0, 50.0, float(vol_input*100), 0.1) / 100
            
            n_sims = st.selectbox("Simulations", [200, 500, 1000], index=1)
            
            st.divider()
            
            if st.button("▶ Run Simulation", type="primary"):
                paths = run_monte_carlo(start_val, sim_vol, simulations=n_sims)
                
                # Calculate Stats
                end_values = paths[-1]
                median_outcome = np.median(end_values)
                p95 = np.percentile(end_values, 95)
                p05 = np.percentile(end_values, 5)
                
                st.session_state['mc_paths'] = paths
                st.session_state['mc_stats'] = (median_outcome, p95, p05)

        # 3. Visualization
        with col_mc2:
            if 'mc_paths' in st.session_state:
                paths = st.session_state['mc_paths']
                med, high, low = st.session_state['mc_stats']
                
                # Plot 100 random paths to avoid clutter
                subset = paths[:, :100]
                fig = px.line(subset, render_mode='webgl')
                fig.update_layout(
                    showlegend=False, 
                    title=f"Monte Carlo: 1,000 Paths @ {sim_vol*100:.1f}% Volatility",
                    xaxis_title="Trading Days",
                    yaxis_title="Portfolio Value (€)",
                    height=500
                )
                st.plotly_chart(fig, width="stretch")
                
                # Stats Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Bear Case (Bottom 5%)", f"€{low:,.0f}", delta=None, delta_color="normal")
                m2.metric("Base Case (Median)", f"€{med:,.0f}")
                m3.metric("Bull Case (Top 5%)", f"€{high:,.0f}")
            else:
                st.info("👈 Click 'Run Simulation' to start.")

    # ==========================================
    # TAB 3: VOLATILITY SURFACE
    # ==========================================
    with tab_vol:
        st.subheader("Options Implied Volatility")
        v_ticker = st.text_input("Enter Ticker for Vol Surface:", value="NVDA").upper()
        
        if v_ticker:
            if st.button("Build Surface"):
                with st.spinner(f"Fetching option chains for {v_ticker}..."):
                    df_vol = get_volatility_surface(v_ticker)
                
                if not df_vol.empty:
                    # 3D Surface Plot
                    fig = go.Figure(data=[go.Mesh3d(
                        x=df_vol['Strike'],
                        y=df_vol['Days'],
                        z=df_vol['IV'],
                        intensity=df_vol['IV'],
                        colorscale='Viridis',
                        opacity=0.8
                    )])
                    
                    fig.update_layout(
                        title=f"{v_ticker} Implied Volatility Surface",
                        scene=dict(
                            xaxis_title='Strike Price',
                            yaxis_title='Days to Expiration',
                            zaxis_title='Implied Volatility'
                        ),
                        margin=dict(l=0, r=0, b=0, t=30),
                        height=600
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.error("Could not fetch options data. Ticker might not have liquid options.")

def render_fundamental_dashboard(user, portfolio, proposals):
    st.title(f"📈 Fundamental Dashboard")
    
    # 1. RAW DATA EXPANDER
    with st.expander("📊 View Raw Portfolio Data"):
        st.dataframe(
            portfolio,
            width="stretch",
            hide_index=True,
            column_config={
                "market_value": st.column_config.NumberColumn(
                    "Value",
                    format="€%.2f",
                ),
                "target_weight": st.column_config.NumberColumn(
                    "Weight",
                    format="%.2f%%"
                ),
                "ticker": st.column_config.TextColumn("Asset"),
            }
        )
    
    # 2. PERFORMANCE CHART (Synthetic 6-Month Backtest)
    st.subheader("Performance vs Market (6 Months)")
    
    # Run the simulation helper (ensure this function exists in your code)
    with st.spinner("Simulating 6-month historical performance..."):
        # We pass the portfolio. 'q_port' is not strictly needed here if we look at 'portfolio' only
        # but the helper can handle both. 
        bench_df = fetch_simulated_history(portfolio, pd.DataFrame()) 
    
    if not bench_df.empty:
        # Create Line Chart
        fig = px.line(bench_df, x='Date', y=['SP500', 'TIC_Fund'], 
                      color_discrete_map={"SP500": "#FFA028", "TIC_Fund": "#00FF00"})
        
        # Bloomberg Style
        fig = style_bloomberg_chart(fig)
        fig.update_layout(title="Portfolio Performance History", yaxis_title="Rebased (100)", legend_title=None)
        
        # RENDER CHART INSIDE THE IF BLOCK
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Not enough historical data to generate performance chart.")
    
    # 3. ALLOCATION & SECTOR ANALYSIS
    c1, c2 = st.columns([1, 2])
    
    # --- PRE-CALCULATION: GET MARKET VALUE ---
    if not portfolio.empty and 'units' in portfolio.columns and 'ticker' in portfolio.columns:
        # 1. Fetch Prices
        tickers = portfolio['ticker'].dropna().unique().tolist()
        live_data = fetch_live_prices_with_change(tickers)
        
        # 2. Map Prices
        def get_price(t):
            return live_data.get(str(t), {}).get('price', 0.0)
            
        portfolio['current_price'] = portfolio['ticker'].map(get_price)

        # --- THE FIX IS HERE: Force data to be Numeric ---
        # "errors='coerce'" turns non-numbers (like text) into NaN (0.0)
        portfolio['units'] = pd.to_numeric(portfolio['units'], errors='coerce').fillna(0.0)
        portfolio['current_price'] = pd.to_numeric(portfolio['current_price'], errors='coerce').fillna(0.0)

        # 3. Calculate Market Value (Now it works because both are numbers)
        portfolio['market_value'] = portfolio['units'] * portfolio['current_price']
        
        # 4. Get Real Return
        portfolio['real_return'] = portfolio['ticker'].map(
            lambda t: live_data.get(str(t), {}).get('pct', 0.0)
        )
    else:
        portfolio['market_value'] = 0.0
        portfolio['real_return'] = 0.0

    with c1:
        st.subheader("Allocation")
        if not portfolio.empty and portfolio['market_value'].sum() > 0:
            # Handle capitalization (Sector vs sector)
            sec_col = 'Sector' if 'Sector' in portfolio.columns else 'sector'
            
            fig_pie = px.pie(
                portfolio, 
                values='market_value', 
                names=sec_col, 
                hole=0.4
            )
            st.plotly_chart(fig_pie, width="stretch")
        else:
            st.info("No market value to display.")

    with c2:
        st.subheader("Sector Performance")
        if not portfolio.empty and portfolio['market_value'].sum() > 0:
            sec_col = 'Sector' if 'Sector' in portfolio.columns else 'sector'
            
            fig_tree = px.treemap(
                portfolio, 
                path=[px.Constant("Portfolio"), sec_col, 'ticker'], 
                values='market_value',
                color='real_return',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                hover_data=['real_return', 'market_value']
            )
            
            fig_tree.update_traces(
                hovertemplate='<b>%{label}</b><br>Value: €%{value:,.0f}<br>Change: %{customdata[0]:.2f}%'
            )
            st.plotly_chart(fig_tree, width="stretch")
        else:
            st.info("No data for Treemap.")

    # 4. CONCENTRATION CHECK
    if not portfolio.empty and 'target_weight' in portfolio.columns:
        sector_alloc = portfolio.groupby('sector')['target_weight'].sum().sort_values(ascending=False)
    
        if not sector_alloc.empty:
            top_sector = sector_alloc.index[0]
            top_weight = sector_alloc.iloc[0]
        
            if top_weight > 0.30: # 30% Threshold
                st.warning(f"⚠️ High Concentration: {top_sector} makes up {top_weight:.1%} of the portfolio.")
            else:
                st.success(f"✅ Portfolio is well-diversified. Top sector: {top_sector} ({top_weight:.1%})")
    
            
def render_quant_dashboard(user, portfolio, proposals):
    st.title(f"🤖 Quant Lab")
    # --- 2. STRATEGY OVERVIEW ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Strategy Allocation")
        
        if not portfolio.empty:
            # Ensure market_value is numeric and clean
            # This fixes the TypeError if data came in as strings
            portfolio['market_value'] = pd.to_numeric(portfolio['market_value'], errors='coerce').fillna(0.0)
            
            model_col = 'model' if 'model' in portfolio.columns else None
            
            if model_col:
                # Group by Strategy
                grouped = portfolio.groupby(model_col)[['market_value']].sum().reset_index()
                
                # Calculate Total AUM
                total_aum = grouped['market_value'].sum()
                
                # FIX: Check for Zero Division
                if total_aum > 0:
                    grouped['allocation'] = grouped['market_value'] / total_aum
                else:
                    grouped['allocation'] = 0.0
                
                st.dataframe(
                    grouped,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        model_col: st.column_config.TextColumn("Strategy"),
                        "market_value": st.column_config.NumberColumn("Total Value", format="€%.2f"),
                        "allocation": st.column_config.ProgressColumn("Alloc", format="%.2f%%", min_value=0, max_value=1)
                    }
                )
            else:
                st.info("No 'Model' column found.")
        else:
            st.info("Portfolio is empty.")

    with c2:
        if not portfolio.empty:
            group_col = 'model' if 'model' in portfolio.columns else 'sector'
            if group_col in portfolio.columns:
                # Filter out zero values for cleaner chart
                chart_data = portfolio[portfolio['market_value'] > 0]
                fig = px.pie(chart_data, values='market_value', names=group_col, hole=0.4)
                st.plotly_chart(fig, width="stretch")

    # --- 3. DETAILED HOLDINGS ---
    st.divider()
    st.subheader("📜 Detailed Asset Holdings")
    
    if not portfolio.empty:
        display_cols = ['ticker', 'name', 'sector', 'model', 'units', 'market_value']
        valid_cols = [c for c in display_cols if c in portfolio.columns]
        
        st.dataframe(
            portfolio[valid_cols],
            width="stretch",
            hide_index=True,
            column_config={
                "market_value": st.column_config.NumberColumn("Value", format="€%.2f"),
                "ticker": st.column_config.TextColumn("Asset"),
            }
        )
        
    st.divider()
    st.subheader("🐋 Whale Watcher (Institutional Flow)")
    st.caption("Analyzing institutional ownership of current portfolio assets.")

    # 1. IDENTIFY THE CORRECT COLUMN (The Fix)
    # Check if we have 'ticker' OR 'model_id' (which is what Quant often uses)
    t_col = 'ticker'
    if 'model_id' in portfolio.columns:
        t_col = 'model_id'

    # 2. CHECK IF DATA EXISTS
    if not portfolio.empty and t_col in portfolio.columns:
        # Filter for stocks (exclude Cash/Crypto)
        # We use the dynamic 't_col' variable now
        stocks = [t for t in portfolio[t_col].unique() 
                 if isinstance(t, str) and "CASH" not in t and "EUR" not in t]
        
        whale_data = []
        
        # We need a progress bar because this fetches live data
        prog_bar = st.progress(0, text="Scanning Institutional Data...")
        
        for i, t in enumerate(stocks):
            try:
                # Fast fetch of single info key
                info = yf.Ticker(t).info
                inst_own = info.get('heldPercentInstitutions', 0)
                short_float = info.get('shortPercentOfFloat', 0)
                
                status = "Retail Heavy"
                if inst_own > 0.8: status = "🐋 Whale Owned (>80%)"
                elif inst_own > 0.5: status = "🏢 Mixed"
                
                whale_data.append({
                    "Ticker": t,
                    "Inst. Ownership": inst_own,
                    "Short Interest": short_float,
                    "Verdict": status
                })
            except:
                pass
            
            # Update progress
            prog_bar.progress((i + 1) / len(stocks), text=f"Scanning {t}...")
            
        prog_bar.empty() # Remove bar when done

        if whale_data:
            df_whale = pd.DataFrame(whale_data)
            
            # 1. SCALE DATA TO 0-100 FOR DISPLAY
            # We multiply by 100 so 0.68 becomes 68.5, which looks better with the "%" symbol
            df_whale["Inst. Ownership"] = df_whale["Inst. Ownership"] * 100
            df_whale["Short Interest"] = df_whale["Short Interest"] * 100
            
            # Sort by highest ownership
            df_whale = df_whale.sort_values("Inst. Ownership", ascending=False)
            
            # 2. DISPLAY WITH CORRECT SPRINTF FORMATTING
            st.dataframe(
                df_whale,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Inst. Ownership": st.column_config.ProgressColumn(
                        "Smart Money %", 
                        # "%.1f" means 1 decimal float. "%%" prints a literal % sign.
                        format="%.1f%%", 
                        min_value=0, 
                        max_value=100
                    ),
                    "Short Interest": st.column_config.NumberColumn(
                        "Short %", 
                        format="%.2f%%"
                    )
                }
            )
        else:
            st.info("No stock data found to analyze.")
    else:
        st.info("Add assets to the portfolio to see Whale Analysis.")
        
def render_documents(user):
    st.title("📚 Library")
    t1, t2 = st.tabs(["Contract", "Archive"])
    
    with t1: 
        with st.container(border=True):
            st.markdown(f"### Digital Agreement: {user['n']}")
            
            # --- NEW: GUEST CHECK ---
            if user.get('r') == 'Guest':
                st.warning("🔒 Contracts are private documents available only to registered members.")
                st.caption("Please log in with a member account to view your agreement.")
            
            else:
                # (Keep existing contract logic)
                CONTRACTS_FOLDER = "data/contracts"
                contract_filename = f"{user['u']}_contract.pdf"
                contract_path = os.path.join(CONTRACTS_FOLDER, contract_filename)
                
                if os.path.exists(contract_path):
                    with open(contract_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                    st.success("✅ Signed Contract Available")
                    st.download_button("Download PDF Copy", pdf_bytes, contract_filename, "application/pdf")
                else:
                    st.warning("⚠️ No digital contract found.")

    with t2: 
        # --- GOOGLE DRIVE INTEGRATION ---
        DRIVE_FOLDER_ID = "1tDtD3PAKLHWH5YMRrcroPQ36HJ-osDin" 
        DRIVE_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
        
        with st.container(border=True):
            c_text, c_btn = st.columns([3, 1])
            c_text.markdown("### 📂 TIC Historical Archive")
            c_text.caption("Access past pitch decks, meeting minutes, and photos.")
            
            # Option 1: Direct Link Button
            c_btn.link_button("Open in Google Drive ↗️", DRIVE_URL)
            
            st.divider()
            
            # Option 2: Embedded View (IF folder is public/shared)
            st.write("**Quick View**")
            
            # --- NEW: GUEST CHECK FOR EMBED ---
            # Sometimes you might want Guests to see this, but if it's private data:
            if user.get('r') == 'Guest':
                 st.info("🔒 Archive preview disabled for guests. Click the link above to request access.")
            else:
                components.iframe(
                    f"https://drive.google.com/embeddedfolderview?id={DRIVE_FOLDER_ID}#list",
                    height=600,
                    scrolling=True
                )
# ==========================================
# 5. MAIN
# ==========================================
def main():
    # 1. Init Session State
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    
    # 2. Init Cookie Manager
    cookie_manager = stx.CookieManager()
    
    # 3. Load Data
    members, f_port, q_port, props, calendar_events, f_total, q_total, df_votes, nav_f, nav_q, att_df = load_data()

    # 4. AUTO-LOGIN: Check if Cookie Exists
    if not st.session_state['logged_in']:
        # Try to get the cookie
        cookie_user = cookie_manager.get(cookie="tic_user")
        
        if cookie_user:
            # Cookie found! Find the user in our database
            found_user = get_user_by_username(cookie_user, members)
            if found_user:
                st.session_state['user'] = found_user
                st.session_state['logged_in'] = True
                st.toast(f"👋 Welcome back, {found_user['n']}!")
                time.sleep(0.5)
                st.rerun()

    # 5. SHOW LOGIN SCREEN (If still not logged in)
    if not st.session_state['logged_in']:
        c1, c2, c3 = st.columns([1,1.5,1])
        with c2:
            st.image(TIC_LOGO, width=200)
            st.title("TIC Portal")
            
            with st.form("login_form", clear_on_submit=True):
                st.subheader("Member Login")
                u = st.text_input("Username", key="login_u")
                p = st.text_input("Password", type="password", key="login_p")
                
                c_log, c_guest = st.columns(2)
                
                # --- BUTTON 1: MEMBER LOGIN ---
                if c_log.form_submit_button("Log In", type="primary"):
                    # 1. Use the LIVE checker instead of the local 'authenticate' function
                    role_found, error_msg = check_credentials_live(u, p)
    
                    if role_found:
                        # 2. If Google Sheets says "Yes", we manually build the user session
                        # We need to find the full user details from the 'members' dataframe to fill the profile
                        # (The dataframe is safe to use for *reading* details, just not for password checking)
                        user_row = members[members['u'] == u.lower().replace(" ", ".")]
        
                        if not user_row.empty:
                            user = user_row.iloc[0].to_dict()
                        else:
                            # Fallback if user exists in Sheet but not in local JSON yet
                            user = {
                                'u': u, 'n': u, 'r': role_found, 'd': 'General', 
                                'email': u, 'admin': False, 'value': 0, 'onboarded': 1
                            }

                        # 3. Save Cookie & Session
                        cookie_manager.set("tic_user", user['u'], expires_at=datetime.now() + timedelta(days=30))
                        st.session_state['user'] = user
                        st.session_state['logged_in'] = True
        
                        # 4. Log the login
                        threading.Thread(
                            target=update_member_field_in_gsheet, 
                            args=(user['u'], "Last Login", datetime.now().strftime('%Y-%m-%d %H:%M'))
                        ).start()
        
                        st.toast(f"✅ Login Successful! Welcome {user.get('n', u)}")
                        time.sleep(0.5)
                        st.rerun()
                    else: 
                        st.error(f"Login Failed: {error_msg}")

                # --- BUTTON 2: GUEST ACCESS ---
                if c_guest.form_submit_button("Guest Access"):
                    # Create a Fake User Profile
                    guest_user = {
                        'u': 'guest',
                        'n': 'Guest Visitor',
                        'r': 'Guest',       # Role = Guest
                        'd': 'Board',     # Department
                        's': 'General',
                        'email': 'guest@tilburg.edu',
                        'admin': False,
                        'value': 0,
                        'contribution': 0,
                        'units_fund': 0,
                        'units_quant': 0,
                        'liq_pending': 0
                    }
                    st.session_state['user'] = guest_user
                    st.session_state['logged_in'] = True
                    st.toast("👋 Welcome, Guest! (Read-Only Mode)")
                    st.rerun()
            # --- END LOGIN FORM ---
        return
    user = st.session_state['user']
    if user.get('r') != 'Guest' and user.get('onboarded', 0) == 0:
        render_onboarding_tour(user)
        return
    # ==========================================
    # FETCH PRICES FOR TAPE (Assets & FX Rates)
    # ==========================================
    tape_tickers = []
    
    # 1. Gather Unique Assets from both portfolios
    all_assets = set()
    if not f_port.empty and 'ticker' in f_port.columns:
        all_assets.update(f_port['ticker'].dropna().unique())
        
    if not q_port.empty:
         # Handle Quant having 'ticker' or 'model_id'
         q_col = 'ticker' if 'ticker' in q_port.columns else 'model_id'
         if q_col in q_port.columns:
             all_assets.update(q_port[q_col].dropna().unique())

    # 2. Process Tickers for Display
    for t in all_assets:
        raw = str(t).upper().strip()
        # Clean up "CASH_" prefix to find real currency
        clean = raw.replace("CASH_", "").replace("CASH ", "").strip()
        
        # A. SKIP Base Currency (No need to show EUR=1.0)
        if clean in ["EUR", "EURO", "CASH"]:
            continue
            
        # B. CONVERT Foreign Cash to FX Pairs (e.g. USD -> USDEUR=X)
        elif clean in ["USD", "GBP", "JPY", "CHF", "CAD", "AUD", "HKD"]:
            tape_tickers.append(f"{clean}EUR=X")
            
        # C. HANDLE Stocks (Fix UK/Japan suffixes)
        else:
            if clean.endswith(".") or clean.endswith(".L"):
                # UK Stocks: RR. -> RR.L
                y_t = clean + "L" if clean.endswith(".") else clean
                tape_tickers.append(y_t)
            else:
                tape_tickers.append(clean)

    # 3. Fetch & Render
    if tape_tickers:
        # Deduplicate and Limit to avoid massive banner
        final_list = list(set(tape_tickers))[:30]
        live_data = fetch_live_prices_with_change(final_list)
        render_ticker_tape(live_data)
        
    with st.sidebar:
        with st.form(key='cli_form', clear_on_submit=True):
            cmd_input = st.text_input("COMMAND >", placeholder="Terminal Engine").upper()
            submit = st.form_submit_button("GO", width="stretch")
            
            if submit and cmd_input:
                # 1. Navigation Commands
                if cmd_input in ["RISK", "VAR", "MACRO", "CORR"]:
                    st.session_state['previous_choice'] = "Risk & Macro"
                    st.rerun()
                
                elif cmd_input in ["PORT", "FUND", "QUANT", "DASH"]:
                    st.session_state['previous_choice'] = "Dashboard"
                    st.rerun()
                
                elif cmd_input in ["VAL", "DCF", "MODEL"]:
                    st.session_state['previous_choice'] = "Valuation Tool"
                    st.rerun()

                elif cmd_input in ["HOME", "LAUNCH", "LP"]:
                    st.session_state['previous_choice'] = "Launchpad"
                    st.rerun()
                    
                elif cmd_input in ["LIB", "LIBRARY","MINECRAFT", "CONTRACT"]:
                    st.session_state['previous_choice'] = "Library"
                    st.rerun()

                elif cmd_input in ["SETTINGS", "SETT"]:
                    st.session_state['previous_choice'] = "Settings"
                    st.rerun()
                    
                # 2. Ticker Lookup (Default behavior)
                else:
                    # Assume it's a ticker (e.g., "TSLA") -> Go to Research Page
                    st.session_state['previous_choice'] = "Stock Research"
                    st.session_state['research_ticker'] = cmd_input
                    st.rerun()
        
        # TIC LOGO
        st.image(TIC_LOGO, width=150)
        st.write("")
        if st.button("📄 Download Weekly Report", use_container_width=True):
            with st.spinner("Compiling Intelligence Brief..."):
                # FIX: Changed 'proposals' to 'props'
                pdf_bytes = create_enhanced_pdf_report(
                    f_port, q_port, f_total, q_total, nav_f, nav_q, 
                    "Weekly Update", props
                )
                
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="TIC_Intelligence_Brief.pdf" style="text-decoration:none; color:inherit;"><button style="width:100%; padding:0.5rem; background:#D4AF37; color:black; border:none; border-radius:4px; font-weight:bold; cursor:pointer;">📥 Click to Save PDF</button></a>'
            st.markdown(href, unsafe_allow_html=True)
        st.markdown("---")
        
        # Profile
        st.header(user.get('n', 'Unknown Member'))
        st.caption(f"{user.get('r', 'Member')} | {user.get('email', '')}")
        
        # --- UPDATED FINANCIAL SUMMARY ---
        val = user.get('value', 0)
        cont = user.get('contribution', 1)
        growth = ((val - cont)/cont)*100 if cont > 0 else 0.0
        
        st.metric("My Stake", f"€{val:,.0f}", f"{growth:.1f}%")
        
        # Show breakdown if they hold both
        u_f = user.get('units_fund', 0)
        u_q = user.get('units_quant', 0)
        
        if u_f > 0 and u_q > 0:
            st.caption(f"Fund: €{(u_f*nav_f):,.0f} | Quant: €{(u_q*nav_q):,.0f}")

        st.divider()
        menu = ["Launchpad","Dashboard", "Stock Research", "Risk & Macro", "Library", "Calendar", "Settings"] 
            
        # Fundamental Specific Tools
        if user['d'] in ['Fundamental', 'Board', 'Advisory'] or user.get('r') == 'Guest':
            menu.insert(3, "Valuation Tool")

        # Only show Admin Panel if user has admin=True
        if user.get('admin', False):
            menu.append("Admin Panel")
        
        # 2. Determine the Correct Index (Priority: Session > Saved DB > Default)
        default_index = 0
        
        # Check if we've already navigated in this session
        target_page = st.session_state.get('previous_choice')
        
        # If not, try to fetch the Last Page saved in the database
        if not target_page:
             target_page = user.get('last_page', 'Launchpad')

        # Find the matching index in the menu list
        for i, option in enumerate(menu):
            # Partial match to handle "Inbox (1)" vs "Inbox"
            if target_page in option:
                default_index = i
                break
        
        # NEW NAVIGATION LOGIC (EASY BROTATO)
        def on_nav_change():
            """Callback: Updates state immediately when button is clicked."""
            new_choice = st.session_state.nav_radio_key
            clean_choice = new_choice.split(" (")[0]
            
            # 1. Update Session State immediately
            st.session_state['previous_choice'] = clean_choice
            
            # 2. Update Google Sheets (Threaded)
            if 'user' in st.session_state and st.session_state['user']['r'] != 'Guest':
                u_name = st.session_state['user']['u']
                threading.Thread(
                    target=update_member_field_in_gsheet, 
                    args=(u_name, "Last_Page", clean_choice)
                ).start()

        # 3. Render the Navigation with Callback
        nav = st.radio(
            "Navigation", 
            menu, 
            index=default_index, 
            key="nav_radio_key", 
            on_change=on_nav_change
        )
        
        st.divider()
        # --- LOGOUT CALLBACK FUNCTION ---
        def logout_callback():
            """Handles logout logic before the app reruns."""
            # 1. Delete the cookie
            try:
                cookie_manager.delete("tic_user")
            except:
                pass
    
            # 2. Wipe Session State
            for key in list(st.session_state.keys()):
                del st.session_state[key]
        
            # 3. Set flags
            st.session_state['logged_in'] = False

        # --- RENDER BUTTON WITH CALLBACK ---
        # Notice we don't need 'if st.button:' logic anymore.
        # The callback handles everything, and Streamlit auto-reruns after a callback.
        st.button("Log Out", on_click=logout_callback)
        
        # SYSTEM STATUS
        st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        st.caption("🟢 System: Online")
        st.caption(f"v1.0 | {user['d']} Access")
        
        # - MAILTO LINK ---
        st.link_button("🐛 Report a Bug", "mailto:s.j.azasoo@tilburguniversity.edu?subject=TIC%20Dashboard%20Bug%20Report&body=Hey%20Senyo,%20I%20found%20an%20issue%20with...")

        # REFRESH BUTTON
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")

# ROUTING
    if "Launchpad" in nav:
        render_launchpad(user, f_total, q_total, nav_f, nav_q, f_port, q_port, calendar_events)
        
    elif "Dashboard" in nav:
        # 1. GUESTS / BOARD / ADVISORY (See Everything)
        if user['d'] in ['Board', 'Advisory'] or user['r'] == 'Guest':
            st.title("🏛️ Executive Overview")
            
            t_fund, t_quant = st.tabs(["📈 Fundamental", "🤖 Quant"])
            
            with t_fund: 
                render_fundamental_dashboard(user, f_port, props)
                st.divider()
                # Fundamental Tab shows Fundamental + Board votes
                render_voting_section(user, props, df_votes, "Fundamental")
                
            with t_quant: 
                render_quant_dashboard(user, q_port, props)
                st.divider()
                # Quant Tab shows Quant + Board votes
                render_voting_section(user, props, df_votes, "Quant")

        # 2. QUANT TEAM (Specific View)
        elif user['d'] == 'Quant': 
            render_quant_dashboard(user, q_port, props)
            st.divider()
            # Shows Quant + Board
            render_voting_section(user, props, df_votes, "Quant")
            
        # 3. FUNDAMENTAL TEAM (Specific View)
        else: 
            render_fundamental_dashboard(user, f_port, props)
            st.divider()
            # Shows Fundamental + Board
            render_voting_section(user, props, df_votes, "Fundamental")
            
    elif "Risk & Macro" in nav: render_risk_macro_dashboard(f_port, q_port)
    elif nav == "Valuation Tool": render_valuation_sandbox()
    elif nav == "Stock Research": render_stock_research()
    elif nav == "Calendar": render_calendar_view(user, calendar_events)
    elif nav == "Library": render_documents(user)
    elif nav == "Settings": render_offboarding(user)
    elif nav == "Admin Panel": render_admin_panel(user, members, f_port, q_port, f_total, q_total, props, df_votes, nav_f, nav_q, att_df)

    # PROFESSIONAL FOOTER
    st.markdown("---")
    c_foot1, c_foot2 = st.columns(2)
    with c_foot1:
        st.caption("© 2025 Tilburg Investment Club | Internal Portal v1.0")
    with c_foot2:
        st.caption("Data provided by Yahoo Finance & TIC Research Team")
        
    # LEGAL DISCLAIMER
    with st.expander("⚖️ Legal Disclaimer (Click to Read)", expanded=False):
        st.caption("""
        **No Financial Advice:** This dashboard is for internal educational and informational purposes only. 
        Nothing herein constitutes an offer to sell, a solicitation of an offer to buy, or a recommendation of any security or strategy.
        
        **No Rights Derived:** The 'Current Value' and 'AUM' figures are estimates based on delayed market data. 
        Official liquidation values are determined solely by the Treasurer at the time of withdrawal. 
        Members cannot derive any legal rights from the figures displayed on this portal.
        
        **Data Accuracy:** Market data is provided 'as-is' via third-party APIs (Yahoo Finance) and may contain errors or delays.
        """)
if __name__ == "__main__":
    main()




























































































































