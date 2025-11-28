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

TIC_LOGO = "https://media.licdn.com/dms/image/v2/D4D0BAQEBPgrthbI7xQ/company-logo_200_200/B4DZoGPeNMJIAI-/0/1761041311048/tilburginvestmentclub_logo?e=1765411200&v=beta&t=cQ2EYv-uszLRoEOqyTbDj_-k9kwcle3ZIos4jcMdq9Q"

st.markdown("""
    <style>
        .stDeployButton {display:none;}
        
        /* FIX: Increased padding to stop titles getting cut off */
        .block-container {
            padding-top: 3rem; 
            padding-bottom: 2rem;
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

@st.cache_data(ttl=3600*12) # Cache for 12 hours since earnings dates don't change often
def fetch_company_events(tickers):
    """Fetches upcoming earnings dates for a list of tickers."""
    events = []
    # Limit to top 15 tickers to prevent app timeout during loading
    safe_tickers = tickers[:15] if tickers else []
    
    for t in safe_tickers:
        try:
            if not isinstance(t, str): continue
            
            # Fetch calendar data
            stock = yf.Ticker(t)
            cal = stock.calendar
            
            # yfinance returns a dict where 'Earnings Date' is a list of dates
            # We look for the 'Earnings Date' key
            if cal and 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                if dates:
                    # Get the next scheduled date (usually index 0)
                    next_date = dates[0]
                    
                    # Ensure we only show future or very recent events
                    if next_date >= date.today():
                        events.append({
                            'title': f"{t} Earnings",
                            'ticker': t,
                            'date': next_date.strftime('%Y-%m-%d'),
                            'type': 'market',
                            'audience': 'all'
                        })
        except Exception:
            # Silently fail for individual tickers if data is missing
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

@st.cache_data(ttl=300)  # Cache for 5 mins to prevent overload
def fetch_live_prices_with_change(tickers):
    """
    Fetches Price AND Hourly Change for all tickers.
    Refreshes only once per hour to save API limits.
    """
    if not tickers: return {}
    
    # Clean tickers: Ensure strings, uppercase, no empty values
    clean_tickers = list(set([
        str(t).upper().strip() for t in tickers 
        if isinstance(t, str) and t and "CASH" not in t.upper()
    ]))
    
    if not clean_tickers: return {}

    try:
        # Fetch 1 day of data to calculate change
        # Threads=True speeds up the download significantly
        data = yf.download(
            clean_tickers, 
            period="1d", 
            interval="1h", 
            group_by='ticker', 
            progress=False,
            threads=True
        )
        
        results = {}
        is_multi = len(clean_tickers) > 1
        
        for t in clean_tickers:
            try:
                # Handle Multi-Index vs Single Index returns from yfinance
                df = data[t] if is_multi else data
                
                # Get valid close prices
                valid_rows = df['Close'].dropna()
                
                if not valid_rows.empty:
                    current_price = float(valid_rows.iloc[-1])
                    
                    # Calculate hourly change (delta)
                    if len(valid_rows) >= 2:
                        prev_price = float(valid_rows.iloc[-2])
                        change = current_price - prev_price
                        pct_change = (change / prev_price) * 100 if prev_price != 0 else 0.0
                    else:
                        change = 0.0
                        pct_change = 0.0
                        
                    results[t] = {
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'pct': round(pct_change, 2)
                    }
                else:
                    results[t] = {'price': 0.0, 'change': 0.0, 'pct': 0.0}
                    
            except Exception:
                results[t] = {'price': 0.0, 'change': 0.0, 'pct': 0.0}
                
        return results
        
    except Exception as e:
        print(f"Price Fetch Error: {e}")
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
    
@st.cache_data(ttl=1800)
def load_static_data():
    """
    Fetches Portfolios and Members using STAGGERED PARALLELISM.
    Fast, but polite to the API.
    """
    client = init_connection()
    if not client: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    sheets_to_fetch = ["Fundamentals", "Quant", "Members", "Events"]
    results = {}
    
    # We use a ThreadPool to run downloads at the same time
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {}
        
        for name in sheets_to_fetch:
            # Submit the task
            future = executor.submit(_fetch_single_sheet, client, name)
            future_map[future] = name
            
            # --- THE FIX: SMALL DELAY ---
            # Wait 200ms before starting the next download.
            # This prevents 4 hits hitting Google at the exact same millisecond.
            time.sleep(0.2) 
            # ----------------------------

        # Collect results as they finish
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as e:
                # print(f"Failed to load {name}: {e}")
                results[name] = pd.DataFrame()

    # Ensure we return safe defaults if a sheet failed
    return (
        results.get("Fundamentals", pd.DataFrame()), 
        results.get("Quant", pd.DataFrame()), 
        results.get("Members", pd.DataFrame()), 
        results.get("Events", [])
    )

@st.cache_data(ttl=60)
def load_dynamic_data():
    """
    Fetches dynamic data using STAGGERED PARALLELISM.
    """
    client = init_connection()
    if not client: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sheets_to_fetch = ["Proposals", "Votes", "Attendance"]
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {}
        for name in sheets_to_fetch:
            future = executor.submit(_fetch_single_sheet, client, name)
            future_map[future] = name
            time.sleep(0.2) # Stagger

        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception:
                results[name] = pd.DataFrame()

    return (
        results.get("Messages", pd.DataFrame()), 
        results.get("Proposals", pd.DataFrame()), 
        results.get("Votes", pd.DataFrame()), 
        results.get("Attendance", pd.DataFrame())
    )
    
def load_data():
    # Master Load Function: Combines Static and Dynamic data
    # 1. Fetch Data (Parallelized & Split)
    f_port_raw, q_port_raw, df_mem, evts = load_static_data()
    msgs, df_props, df_votes, att_raw = load_dynamic_data()
    # 2. Initialize Defaults
    members = pd.DataFrame()
    f_port = pd.DataFrame(); q_port = pd.DataFrame()
    messages = []; proposals = []; full_calendar = []
    f_total = 0.0; q_total = 0.0
    nav_fund = 100.00; nav_quant = 100.00

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
        df_mem.columns = df_mem.columns.astype(str).str.strip()
        
        # A. Sum Total Units Outstanding (The Denominator)
        # Ensure we treat empty strings as 0
        total_units_fund = pd.to_numeric(df_mem.get('Units_Fund', 0), errors='coerce').fillna(0).sum()
        total_units_quant = pd.to_numeric(df_mem.get('Units_Quant', 0), errors='coerce').fillna(0).sum()

        # B. Calculate Live NAV (Net Asset Value per Share)
        # Logic: Total Assets / Total Shares
        if total_units_fund > 0:
            nav_fund = f_total / total_units_fund
        
        if total_units_quant > 0:
            nav_quant = q_total / total_units_quant

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
            
            try: liq_val = int(float(row.get('Liq Pending', 0)))
            except: liq_val = 0
            
            last_active = str(row.get('Last Login', 'Never'))
            last_p = str(row.get('Last_Page', 'Launchpad'))

            members_list.append({
                'u': uname, 'p': str(row.get('Password', 'pass')).strip(), 'n': name, 'email': email,
                'r': role_data['r'], 'd': role_data['d'], 's': role_data['s'], 
                'admin': role_data.get('admin', False),
                'status': 'Pending' if liq_val == 1 else 'Active', 'liq_pending': liq_val,
                'contribution': clean_float(row.get('Initial Investment', 0)),
                'value': real_value, # <--- This is now dynamically calculated
                'units_fund': u_f,
                'units_quant': u_q,
                'last_login': last_active, 'last_page': last_p,
            })
        members = pd.DataFrame(members_list)
    else:
        members = pd.DataFrame([{'u': 'admin', 'p': 'pass', 'n': 'Offline Admin', 'r': 'Admin', 'd': 'Board', 'admin': True, 'value': 0}])

    # 5. Process Events
    manual_events = []
    if not evts.empty:
        evts['Date'] = pd.to_datetime(evts['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        for _, row in evts.iterrows():
            manual_events.append({
                'title': str(row.get('Title','')), 'ticker': str(row.get('Ticker','')),
                'date': str(row.get('Date','')), 'type': str(row.get('Type','')).lower(),
                'audience': str(row.get('Audience','all'))
            })

    real_events = []
    if not f_port.empty and 'ticker' in f_port.columns:
        valid_tickers = [t for t in f_port['ticker'].dropna().unique() if isinstance(t,str) and "CASH" not in t]
        if valid_tickers:
            real_events = fetch_company_events(valid_tickers)
    
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
        att['Date'] = pd.to_datetime(att['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

    return members, f_port, q_port, proposals, full_calendar, f_total, q_total, df_votes, nav_fund, nav_quant, att
# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600*12)
def fetch_simulated_history(f_port, q_port):
    """
    Creates a synthetic 6-month backtest.
    FIXED: Handles yfinance Multi-Index column outputs.
    """
    # 1. Prepare Data
    combined = pd.concat([f_port, q_port], ignore_index=True)
    
    # Setup Fallback (Dummy Data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    dummy_df = pd.DataFrame({'Date': dates, 'SP500': 100.0, 'TIC_Fund': 100.0})

    if combined.empty:
        return dummy_df

    # --- SAFETY: Force Numeric Conversion ---
    if 'market_value' in combined.columns:
        combined['market_value'] = pd.to_numeric(combined['market_value'], errors='coerce').fillna(0.0)
    else:
        return dummy_df 
        
    # 2. Filter for Non-Cash Assets
    # Exclude currencies/cash from the stock curve
    is_cash = combined['ticker'].astype(str).str.upper().isin(["CASH", "EUR", "EURO", "USD", "GBP", "JPY", "CHF"])
    is_cash = is_cash | combined['ticker'].astype(str).str.upper().str.startswith("CASH")
    
    equity_df = combined[~is_cash].copy()
    
    # If portfolio is 100% cash, return flat line
    if equity_df.empty or equity_df['market_value'].sum() == 0:
        return dummy_df

    # 3. Calculate Weights
    total_equity_val = equity_df['market_value'].sum()
    equity_df['weight'] = equity_df['market_value'] / total_equity_val
    
    # 4. Identify Tickers to Fetch
    fetch_map = {}
    for t in equity_df['ticker'].unique():
        clean_t = str(t).upper().strip()
        
        # Handle UK (.L) and Japan (.T)
        if clean_t.endswith(".") or clean_t.endswith(".L"):
            y_t = clean_t + "L" if clean_t.endswith(".") else clean_t
            fetch_map[clean_t] = y_t
        elif clean_t.endswith(".T"):
             fetch_map[clean_t] = clean_t 
        else:
            fetch_map[clean_t] = clean_t
            
    tickers_to_download = list(set(fetch_map.values()))
    
    # Ensure GSPC is in the download list
    if "^GSPC" not in tickers_to_download:
        tickers_to_download.append("^GSPC")

    # 5. Download History
    try:
        # Fetch data
        data_raw = yf.download(tickers_to_download, period="6mo", progress=False)
        
        # If download failed completely
        if data_raw.empty:
            return dummy_df

        # --- CRITICAL FIX: Flatten Multi-Index Columns ---
        # yfinance often returns columns like ('Close', 'AAPL'). We want just 'AAPL'.
        if isinstance(data_raw.columns, pd.MultiIndex):
            # If 'Close' is in the levels, grab just the Close price
            try:
                data = data_raw['Close'].copy()
            except KeyError:
                # Fallback: just take the last level (Ticker)
                data = data_raw.copy()
                data.columns = data.columns.get_level_values(-1)
        else:
            # It's already flat (single ticker download sometimes does this)
            # If it's a single column 'Close', we need to rename it to the ticker
            if len(tickers_to_download) == 1 and 'Close' in data_raw.columns:
                 data = data_raw[['Close']].rename(columns={'Close': tickers_to_download[0]})
            elif 'Close' in data_raw.columns:
                 data = data_raw['Close'].copy()
            else:
                 data = data_raw.copy()

        # Fill Gaps
        data = data.ffill().bfill().dropna()
        
        if data.empty: 
            return dummy_df
        
        # 6. Normalize Data (Start at 100)
        # Ensure Benchmark exists
        if '^GSPC' not in data.columns:
            # If SPY failed, use the first available column as a proxy or flat line
            data['^GSPC'] = 100.0

        normalized = (data / data.iloc[0]) * 100
        
        # 7. Construct Portfolio Curve
        portfolio_curve = pd.Series(0.0, index=normalized.index)
        
        # We assume 100% equity allocation for the curve (rebased to 100)
        # Iterate over our fetch_map to match Sheet Ticker -> YFinance Column
        matched_weight = 0.0
        
        for raw_t, y_t in fetch_map.items():
            if y_t in normalized.columns:
                w = equity_df.loc[equity_df['ticker'] == raw_t, 'weight'].sum()
                portfolio_curve += normalized[y_t] * w
                matched_weight += w
            else:
                # If ticker missing in Yahoo data, add its weight as flat Cash (100)
                # This prevents the curve from dropping if a ticker fails
                w = equity_df.loc[equity_df['ticker'] == raw_t, 'weight'].sum()
                portfolio_curve += 100.0 * w
        
        # 8. Final DataFrame
        df_chart = pd.DataFrame({
            'Date': normalized.index,
            'SP500': normalized['^GSPC'],
            'TIC_Fund': portfolio_curve
        })
        
        return df_chart.reset_index(drop=True)
        
    except Exception as e:
        print(f"History Simulation Error: {e}")
        return dummy_df
        
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
    user = df[(df['u'] == username) & (df['p'] == password)]
    return user.iloc[0] if not user.empty else None

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
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- 1. EXECUTIVE SUMMARY ---
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Executive Summary: {report_title}", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)
    
    # Financial Metrics
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 8, "Metric", 1, 0, 'L', 1)
    pdf.cell(60, 8, "Value (EUR)", 1, 1, 'R', 1)
    pdf.set_font("Arial", "", 10)
    pdf.cell(60, 8, "Total AUM", 1, 0, 'L')
    pdf.cell(60, 8, f"{f_total + q_total:,.2f}", 1, 1, 'R')
    pdf.cell(60, 8, "Fundamental Fund", 1, 0, 'L')
    pdf.cell(60, 8, f"{f_total:,.2f}", 1, 1, 'R')
    pdf.cell(60, 8, "Quant Fund", 1, 0, 'L')
    pdf.cell(60, 8, f"{q_total:,.2f}", 1, 1, 'R')
    pdf.ln(5)

    # --- 2. MARKET CONTEXT (New) ---
    # We pull this live to give context to the AUM numbers
    macro = fetch_macro_data() 
    vix = macro.get('VIX', {}).get('value', 0)
    yield_10y = macro.get('10Y Treasury', {}).get('value', 0)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Market Context", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Market Volatility (VIX): {vix:.2f} | US 10Y Yield: {yield_10y:.2f}%", ln=True)
    pdf.ln(5)

    # --- 3. GOVERNANCE & STRATEGY (New) ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Governance & Strategic Decisions", ln=True)
    
    # Filter Proposals
    applied_props = [p for p in proposals if str(p.get('Applied')) == '1']
    active_props = [p for p in proposals if str(p.get('Applied')) == '0']
    
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 8, "Recently Executed / Applied:", ln=True)
    pdf.set_font("Arial", "", 9)
    
    if applied_props:
        for p in applied_props:
            # Format: [Fundamental] BUY: AMD
            line = f"[{p.get('Dept')}] {p.get('Type')}: {p.get('Item')} - {p.get('Description')[:60]}..."
            # Draw a small green checkmark (using ASCII or simplified char)
            pdf.cell(5, 6, "x", 0, 0) 
            pdf.cell(0, 6, line, ln=True)
    else:
        pdf.cell(0, 6, "No recent executed proposals.", ln=True)
        
    pdf.ln(2)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 8, "Active Voting Items:", ln=True)
    pdf.set_font("Arial", "", 9)
    
    if active_props:
        for p in active_props:
            line = f"[{p.get('Dept')}] {p.get('Type')}: {p.get('Item')} (Ends: {p.get('End_Date')})"
            pdf.cell(5, 6, "-", 0, 0)
            pdf.cell(0, 6, line, ln=True)
    else:
        pdf.cell(0, 6, "No active voting items.", ln=True)
    
    pdf.ln(10)

    # --- 4. PORTFOLIO SNAPSHOTS ---
    # (Keep your existing Holdings tables here...)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Top Fundamental Holdings", ln=True)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(30, 8, "Ticker", 1, 0, 'L', 1)
    pdf.cell(80, 8, "Asset Name", 1, 0, 'L', 1)
    pdf.cell(40, 8, "Weight", 1, 1, 'R', 1)
    
    pdf.set_font("Arial", "", 9)
    if not f_port.empty and 'target_weight' in f_port.columns:
        top_f = f_port.sort_values('target_weight', ascending=False).head(8)
        for _, row in top_f.iterrows():
            name = str(row.get('name', 'Unknown'))[:35]
            ticker = str(row.get('ticker', ''))
            weight = float(row.get('target_weight', 0)) * 100
            pdf.cell(30, 7, ticker, 1, 0, 'L')
            pdf.cell(80, 7, name, 1, 0, 'L')
            pdf.cell(40, 7, f"{weight:.1f}%", 1, 1, 'R')
    
    return pdf.output(dest='S').encode('latin-1')
    
# ==========================================
# 4. VIEW COMPONENTS
# ==========================================
def render_launchpad(user, f_total, q_total, nav_f, nav_q, f_port, q_port, calendar_events):
    """
    Role-Based Homepage with REAL Data Connections.
    FIXED: Added type conversion to prevent math errors on string data.
    """
    # 1. Dynamic Greeting
    h = datetime.now().hour
    if 5 <= h < 12: greeting = "Good Morning"
    elif 12 <= h < 18: greeting = "Good Afternoon"
    else: greeting = "Good Evening"
    
    st.title(f"🚀 {greeting}, {user['n']}")
    st.caption(f"Logged in as: {user['r']} | Department: {user['d']}")
    if user.get('r') == 'Guest':
        st.info(
            "**Welcome to the Tilburg Investment Club portal.**\n\n"
            "Feel free to explore our dashboards, risk models, and library.\n\n"
            "**Note:** As a guest, you have read-only access. Voting and trading features are disabled.",
            icon="👋"
        )
    # 2. Global Ticker (Mini)
    c1, c2, c3, c4 = st.columns(4)
    # Calc (short for calculator) Total TIC AUM + TOP DIFF
    safe_delta = float(total_delta_eur)
    
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
                active_models = q_port['model_id'].nunique()
            
            # Ensure market_value is numeric for sorting
            q_port['mv_numeric'] = pd.to_numeric(q_port['market_value'], errors='coerce').fillna(0)
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
            
def render_voting_section(user, proposals, votes_df, target_dept):
    """Renders the voting UI with a Circular Donut Chart."""
    st.header(f"🗳️ {target_dept} Governance")
    
    # Filter active proposals
    active_props = [p for p in proposals if p.get('Dept') == target_dept and str(p.get('Applied')) == '0']
    
    if not active_props:
        st.info("No active proposals.")
        return

    for p in active_props:
        p_id = str(p['ID'])
        
        with st.container(border=True):
            # Layout: Description (Left) | Chart (Middle) | Buttons (Right)
            c_desc, c_chart, c_act = st.columns([3, 1.5, 1.5])
            
            # --- 1. CALCULATE VOTES ---
            if not votes_df.empty:
                votes_df['Proposal_ID'] = votes_df['Proposal_ID'].astype(str)
                relevant_votes = votes_df[votes_df['Proposal_ID'] == p_id]
                yes_count = len(relevant_votes[relevant_votes['Vote'] == 'YES'])
                no_count = len(relevant_votes[relevant_votes['Vote'] == 'NO'])
            else:
                yes_count, no_count = 0, 0
            
            total = yes_count + no_count

            # --- COLUMN 1: DESCRIPTION ---
            with c_desc:
                st.subheader(f"{p.get('Type')}: {p.get('Item')}")
                st.write(p.get('Description'))
                st.caption(f"Ends: {p.get('End_Date')}")

            # --- COLUMN 2: CIRCULAR CHART ---
            with c_chart:
                if total > 0:
                    fig = go.Figure(data=[go.Pie(
                        labels=['For', 'Against'],
                        values=[yes_count, no_count],
                        hole=0.6,
                        marker=dict(colors=['#228B22', '#D2042D']),
                        textinfo='none',
                        hoverinfo='label+value+percent'
                    )])
                    
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=120,
                        annotations=[dict(text=f"{yes_count}v{no_count}", x=0.5, y=0.5, font_size=16, showarrow=False, font=dict(color="white"))],
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    # FIX: Added unique key=f"chart_{p_id}" to prevent Duplicate ID error
                    st.plotly_chart(
                        fig, 
                        use_container_width=True, 
                        config={'displayModeBar': False}, 
                        key=f"chart_{p_id}"
                    )
                else:
                    st.write("") # Spacer
                    st.caption("No votes yet")

            # --- COLUMN 3: ACTIONS ---
            with c_act:
                st.write("")
                
                # Guest check
                if user.get('r') == 'Guest':
                    st.info("🔒 Guest View")
                    st.caption("Voting disabled.")
                
                else:
                    # (Keep your existing voting logic here)
                    user_has_voted = False
                    if not votes_df.empty:
                        user_vote = votes_df[(votes_df['Proposal_ID'] == p_id) & (votes_df['Username'] == user['u'])]
                        if not user_vote.empty: user_has_voted = True
                    
                    if user_has_voted:
                        st.success("✅ Voted")
                    else:
                        c_y, c_n = st.columns(2)
                        if c_y.button("YES", key=f"y_{p_id}"):
                            if cast_vote_gsheet(p_id, user['u'], "YES"): st.success("Voted!"); st.rerun()
                        if c_n.button("NO", key=f"n_{p_id}"):
                            if cast_vote_gsheet(p_id, user['u'], "NO"): st.error("Voted!"); st.rerun()
                    
                    # Admin Execute (Only show if Admin AND Passing)
                    if user.get('admin', False) and total > 0 and yes_count > no_count:
                        if st.button("Execute", key=f"exe_{p_id}"):
                            # ... (Keep existing execute logic) ...
                            client = init_connection()
                            sheet = client.open("TIC_Database_Master")
                            ws = sheet.worksheet("Proposals")
                            cell = ws.find(p_id)
                            if cell:
                                ws.update_cell(cell.row, 7, 1)
                                st.success("Applied!")
                                st.cache_data.clear()
                                st.rerun()
                            
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
        if st.button("Analyze", type="primary", use_container_width=True):
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
                    st.metric("Div Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
                    st.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.1f}%")

    with t2:
        st.subheader("Financial Statements")
        type_select = st.radio("Select View:", ["Income Statement", "Balance Sheet", "Cash Flow"], horizontal=True, label_visibility="collapsed")
        
        df_show = pd.DataFrame()
        if type_select == "Income Statement": df_show = fins['inc']
        elif type_select == "Balance Sheet": df_show = fins['bal']
        else: df_show = fins['cash']
        
        if not df_show.empty:
            st.dataframe(df_show, use_container_width=True, height=500)
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
                    use_container_width=True
                )
            else:
                st.warning("Could not load peer data.")
        else:
            st.info("Click the button above to load peer data (saves API usage).")
            
def render_valuation_sandbox():
    st.title("🧮 Valuation Sandbox (DCF)")
    st.caption("DISCOUNTED CASH FLOW MODEL // INTRINSIC VALUE CALCULATOR")

    c_inputs, c_viz = st.columns([1, 2])
    
    with c_inputs:
        with st.container(border=True):
            st.markdown("### 1. INPUTS")
            ticker = st.text_input("Ticker Symbol", "AAPL")
            fcf = st.number_input("Last FCF ($B)", value=100.0)
            shares = st.number_input("Shares Out (B)", value=15.0)
            cash = st.number_input("Net Cash ($B)", value=50.0)
            
            st.markdown("### 2. ASSUMPTIONS")
            growth_1_5 = st.slider("Growth (Yr 1-5)", 0.0, 0.30, 0.10, 0.01)
            growth_6_10 = st.slider("Growth (Yr 6-10)", 0.0, 0.20, 0.05, 0.01)
            wacc = st.slider("WACC %", 0.05, 0.15, 0.09, 0.005)
            term_growth = st.slider("Terminal Growth %", 0.01, 0.05, 0.025, 0.005)

    # --- CORE CALCULATION FUNCTION ---
    def calculate_dcf(w_in, g_in):
        # Years 1-5
        future_fcf = []
        discount_factors = []
        for i in range(1, 6):
            future_fcf.append(fcf * ((1 + growth_1_5) ** i))
            discount_factors.append((1 + w_in) ** i)
        
        # Years 6-10
        for i in range(1, 6):
            future_fcf.append(future_fcf[-1] * ((1 + growth_6_10) ** i))
            discount_factors.append((1 + w_in) ** (5 + i))
            
        # Terminal Value
        tv = (future_fcf[-1] * (1 + g_in)) / (w_in - g_in)
        pv_tv = tv / ((1 + w_in) ** 10)
        
        pv_fcf = sum([f/d for f, d in zip(future_fcf, discount_factors)])
        
        equity_val = pv_fcf + pv_tv + cash
        return equity_val / shares, pv_fcf, pv_tv

    # Base Case
    share_price, pv_fcf, pv_tv = calculate_dcf(wacc, term_growth)

    with c_viz:
        st.subheader(f"IMPLIED VALUE: ${share_price:.2f}")
        
        # 1. Waterfall Chart
        fig = px.bar(
            x=['PV FCF (10yr)', 'PV Term Val', 'Net Cash'],
            y=[pv_fcf, pv_tv, cash],
            title=f"ENTERPRISE VALUE BRIDGE: {ticker.upper()}",
            labels={'y':'Value ($B)', 'x':''}
        )
        fig.update_traces(marker_color=['#FF9900', '#444444', '#00FF00']) 
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 2. SENSITIVITY TABLE (HEATMAP)
        st.markdown("#### 🌡️ Sensitivity Analysis (Price vs Assumptions)")
        
        # Create Ranges
        wacc_range = np.linspace(wacc - 0.01, wacc + 0.01, 5) # +/- 1%
        term_range = np.linspace(term_growth - 0.005, term_growth + 0.005, 5) # +/- 0.5%
        
        # Build Matrix
        z_values = []
        for w in wacc_range:
            row = []
            for g in term_range:
                p, _, _ = calculate_dcf(w, g)
                row.append(round(p, 2))
            z_values.append(row)
            
        # Plot Heatmap
        fig_heat = px.imshow(
            z_values,
            labels=dict(x="Terminal Growth", y="WACC", color="Share Price"),
            x=[f"{x:.1%}" for x in term_range],
            y=[f"{y:.1%}" for y in wacc_range],
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdYlGn"
        )
        fig_heat.update_layout(title="Implied Share Price Matrix")
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

def render_admin_panel(user, members_df, f_port, q_port, f_total, q_total, proposals, votes_df, nav_f, nav_q, attendance_df):
    st.title("🔒 Admin Console")
    st.info(f"Logged in as: {user['n']} ({user['r']})")
    # TABS
    tabs = ["👥 Member Database", "💰 Treasury", "📄 Reporting", "✅ Attendance", "🗳️ Governance"]
    if 'admin_active_tab' not in st.session_state:
        st.session_state['admin_active_tab'] = tabs[0]
    
    try:
        curr_index = tabs.index(st.session_state['admin_active_tab'])
    except:
        curr_index = 0
        
    active_tab = st.radio("Admin Menu", tabs, index=curr_index, horizontal=True, label_visibility="collapsed")
    
    st.session_state['admin_active_tab'] = active_tab
    st.divider()    
    # --- TAB 1: MEMBER DATABASE (READ/WRITE) ---
    if active_tab == "👥 Member Database":
        st.subheader("Manage Membership")
        st.markdown("Edit roles, emails, or status directly below.")
        if 'members_db' not in st.session_state: st.session_state['members_db'] = members_df
        
        edited_df = st.data_editor(
            st.session_state['members_db'][['n', 'email', 'r', 'd', 's', 'contribution', 'value', 'status', 'last_login']],
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                    "last_login": st.column_config.TextColumn("Last Active", disabled=True),
                    "contribution": st.column_config.NumberColumn("Invested (€)", format="€%.2f"),
                    "value": st.column_config.NumberColumn("Value (€)", format="€%.2f")
                }
        )
        
        if st.button("Save Database Changes"):
            st.session_state['members_db'] = edited_df
            
            # FIX: Save to the specific absolute path
            try:
                edited_df.to_excel(MEMBER_FILE_PATH, index=False, engine='openpyxl')
                st.success(f"Database successfully updated at:\n{MEMBER_FILE_PATH}")
            except Exception as e:
                st.error(f"Failed to save file: {e}. Is the file open in Excel?")

    # --- TAB 2: TREASURY (UNITIZED SYSTEM) ---
    elif active_tab == "💰 Treasury":
        st.subheader("Fund Treasury (Unitized)")
        st.caption("Issue new shares based on current Net Asset Value (NAV).")
        
        # Display Current NAV
        c_nav1, c_nav2, c_nav3 = st.columns(3)
        c_nav1.metric("NAV Fundamental", f"€{nav_f:.2f}")
        c_nav2.metric("NAV Quant", f"€{nav_q:.2f}")
        c_nav3.metric("Total AUM", f"€{f_total + q_total:,.2f}")
        
        st.divider()
        
        with st.expander("💰 Process Capital Injection (Buy Shares)", expanded=True):
            with st.form("issue_units"):
                target_user = st.selectbox("Select Member", members_df['u'].tolist())
                amount = st.number_input("Cash Injected (€)", min_value=0.0, step=50.0)
                
                # Treasurer chooses where the money goes
                fund_choice = st.radio("Allocate to:", ["Fundamental Fund", "Quant Fund", "50/50 Split"])
                
                # Live Preview of Units to be Issued
                preview_f, preview_q = 0.0, 0.0
                if amount > 0:
                    if fund_choice == "Fundamental Fund":
                        preview_f = amount / nav_f
                    elif fund_choice == "Quant Fund":
                        preview_q = amount / nav_q
                    else: # 50/50
                        preview_f = (amount * 0.5) / nav_f
                        preview_q = (amount * 0.5) / nav_q
                
                st.info(f"Est. Issuance: {preview_f:.4f} Fund Units | {preview_q:.4f} Quant Units")
                
                if st.form_submit_button("✅ Issue Units & Update DB"):
                    client = init_connection()
                    sheet = client.open("TIC_Database_Master")
                    ws = sheet.worksheet("Members")
                    
                    # Find user row
                    data = ws.get_all_records()
                    df = pd.DataFrame(data)
                    # Create matching column
                    df['u_match'] = df['Name'].astype(str).str.lower().str.strip().str.replace(' ', '.')
                    
                    try:
                        # Gspread is 1-indexed, plus 1 for header = index + 2
                        idx = df.index[df['u_match'] == target_user].tolist()[0]
                        row_num = idx + 2
                        
                        # Get Header Indices dynamically
                        headers = ws.row_values(1)
                        col_fund = headers.index('Units_Fund') + 1
                        col_quant = headers.index('Units_Quant') + 1
                        col_inv = headers.index('Initial Investment') + 1
                        
                        # Get Current Values
                        curr_f = float(df.iloc[idx]['Units_Fund']) if df.iloc[idx]['Units_Fund'] != '' else 0.0
                        curr_q = float(df.iloc[idx]['Units_Quant']) if df.iloc[idx]['Units_Quant'] != '' else 0.0
                        curr_inv = float(df.iloc[idx]['Initial Investment']) if df.iloc[idx]['Initial Investment'] != '' else 0.0
                        
                        # Update Cells
                        ws.update_cell(row_num, col_fund, curr_f + preview_f)
                        ws.update_cell(row_num, col_quant, curr_q + preview_q)
                        ws.update_cell(row_num, col_inv, curr_inv + amount) # We track investment just for ROI calc
                        
                        st.success(f"Success! Issued {preview_f:.2f} Fund / {preview_q:.2f} Quant units to {target_user}.")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error updating database: {e}")
        
        st.subheader("Club Financials")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cash Balance", "€12,450.00", "+€500")
        c2.metric("Pending Payouts", "€0.00")
        # DYNAMIC AUM: Uses the passed argument
        c3.metric("Total AUM", f"€{f_total + q_total:,.2f}")
        
        st.divider()
        st.subheader("Liquidation Queue")
        
        # 1. Filter and Prepare the Queue DataFrame
        # liq_pending=1 is the core filter
        pending_requests = members_df[members_df['liq_pending'] == 1].copy()
        
        if not pending_requests.empty:
            # Prepare display DataFrame, ensuring 'u' (username) is available for lookup
            liq_df_edit = pending_requests.rename(columns={'n': 'Member', 'value': 'Amount'})
            
            # Add a checkbox column for selection and initialize it to False
            liq_df_edit['Approve'] = False 
            
            # Select the columns needed for the editor display and processing
            liq_df_edit = liq_df_edit[['Approve', 'Member', 'Amount', 'u']]
            
            # Display DataFrame with checkbox enabled
            selected_data = st.data_editor(
                liq_df_edit.drop(columns=['u']), # Drop 'u' for visual display
                hide_index=True,
                column_config={
                    "Approve": st.column_config.CheckboxColumn(default=False),
                    "Amount": st.column_config.NumberColumn(format="€%.2f"),
                },
                use_container_width=True,
                key='liq_queue_editor' # Key for accessing selections
            )
            
            # 3. Identify selected users
            # Get the rows from the output where 'Approve' was ticked
            ticked_rows = selected_data[selected_data['Approve'] == True]
            
            # Map the index of ticked rows back to the 'u' column of the input data ('liq_df_edit')
            selected_usernames = liq_df_edit.loc[ticked_rows.index, 'u'].tolist()

            c_a, c_b = st.columns(2)
            
            # APPROVE BUTTON LOGIC
            if c_a.button(f"Approve {len(selected_usernames)} Request(s)", type="primary", disabled=not selected_usernames):
                # APPROVE: Liq Pending = 0, Liq Approved = 1
                updates = {'Liq Pending': 0, 'Liq Approved': 1}
                # Call the helper function to update the Excel file
                if update_member_fields_in_gsheet_bulk(selected_usernames, updates):
                    st.success(f"✅ Approved {len(selected_usernames)} requests. Please clear cache (or restart) to update.")
                else:
                    st.error("❌ Approval failed. Check file permissions.")
                st.rerun()

            # DENY BUTTON LOGIC
            if c_b.button("Deny Selected", disabled=not selected_usernames):
                # DENY: Liq Pending = 0, Liq Approved = 0 (Clears pending state)
                updates = {'Liq Pending': 0, 'Liq Approved': 0}
                if update_member_fields_in_gsheet_bulk(selected_usernames, updates):
                    st.info(f"Requests for {len(selected_usernames)} users denied and cleared.")
                else:
                    st.error("❌ Denial failed.")
                st.rerun()

        else:
            st.info("✅ No pending liquidation requests.")
        pass

    # --- TAB 3: REPORTING ---
    elif active_tab == "📄 Reporting":
        st.subheader("Generate Official Reports")
        st.caption("Creates a PDF snapshot of the current portfolio state, AUM, and governance log.")
        
        # 1. Inputs (No Form)
        report_title = st.text_input("Report Title", value=f"Status Report - {datetime.now().strftime('%B %Y')}")
        
        # 2. Generate Button
        if st.button("📄 Prepare PDF Report"):
            with st.spinner("Generating PDF..."):
                # Generate and save to Session State
                pdf_bytes = create_enhanced_pdf_report(
                    f_port, q_port, 
                    f_total, q_total, 
                    nav_f, nav_q, 
                    report_title,
                    proposals
                )
                st.session_state['generated_pdf_data'] = pdf_bytes
                st.success("Report Ready!")

        # 3. Download Button (Shows only if report exists in memory)
        if 'generated_pdf_data' in st.session_state:
            fname = f"TIC_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
            
            st.download_button(
                label="📥 Download PDF",
                data=st.session_state['generated_pdf_data'],
                file_name=fname,
                mime="application/pdf"
            )
        st.divider()
        st.subheader("💾 System Backup")
        st.caption("Download raw database snapshots for offline storage.")
    
        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    
        # 1. Members Backup
        c_b1.download_button(
            "👥 Members",
            members_df.to_csv(index=False).encode('utf-8'),
            f"Backup_Members_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    
        # 2. Votes Backup
        if not votes_df.empty:
            c_b2.download_button(
                "🗳️ Votes",
                votes_df.to_csv(index=False).encode('utf-8'),
                f"Backup_Votes_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        
        # 3. Fundamental Portfolio Backup
        c_b3.download_button(
            "📈 Fundamentals",
            f_port.to_csv(index=False).encode('utf-8'),
            f"Backup_Fund_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

        # 4. Quant Portfolio Backup
        c_b4.download_button(
            "🤖 Quant",
            q_port.to_csv(index=False).encode('utf-8'),
            f"Backup_Quant_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )    
        pass
    
    # --- TAB 4: ATTENDANCE ---
    elif active_tab == "✅ Attendance":
        st.subheader("Meeting Attendance Tracker")
        
        # 1. Select Meeting Date
        meet_date = st.date_input("Meeting Date", datetime.now())
        date_str = meet_date.strftime('%Y-%m-%d')
        
        # 2. Interactive Table
        # We create a temporary dataframe for the UI
        input_df = pd.DataFrame({
            'Member': members_df['n'],
            'Username': members_df['u'], # Hidden ID
            'Present': True # Default to Present
        })
        
        edited_att = st.data_editor(
            input_df,
            column_config={
                "Username": None, # Hide this column
                "Present": st.column_config.CheckboxColumn("Present?", default=True)
            },
            hide_index=True,
            use_container_width=True
        )
        
        # 3. Save Button
        if st.button("💾 Save Attendance Log"):
            # Convert the editor data into the format needed for the helper
            att_dict = {}
            for index, row in edited_att.iterrows():
                status = "Present" if row['Present'] else "Absent"
                att_dict[row['Username']] = status
            
            if save_attendance_log(date_str, att_dict):
                st.success(f"Attendance for {date_str} saved successfully.")
            else:
                st.error("Failed to save attendance.")
        
        st.divider()
        
        # 4. View History
        st.markdown("#### 📜 History Log")
        # Placeholder until you update the function signature
        st.dataframe(attendance_df, use_container_width=True)
    
    # --- TAB 5: GOVERNANCE ---
    elif active_tab == "🗳️ Governance":
        st.subheader("Proposal Archive & Live Results")
        
        if not proposals:
            st.info("No proposals found in database.")
        else:
            # 1. Aggregate Votes
            summary_data = []
            for p in proposals:
                p_id = str(p['ID'])
                
                # Filter votes for this specific proposal
                if not votes_df.empty:
                    # Ensure types match for filtering
                    votes_df['Proposal_ID'] = votes_df['Proposal_ID'].astype(str)
                    p_votes = votes_df[votes_df['Proposal_ID'] == p_id]
                    yes = len(p_votes[p_votes['Vote'] == 'YES'])
                    no = len(p_votes[p_votes['Vote'] == 'NO'])
                else:
                    yes, no = 0, 0
                
                # Determine Status Label
                status = "🔴 Applied/Closed" if str(p.get('Applied')) == '1' else "🟢 Active"
                
                summary_data.append({
                    "ID": p_id,
                    "Type": p.get('Type'),
                    "Item": p.get('Item'),
                    "Dept": p.get('Dept'),
                    "✅ Yes": yes,
                    "❌ No": no,
                    "Total": yes + no,
                    "Status": status
                })
            
            # 2. Display as Interactive Table
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(
                df_summary, 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn(
                        "Status", 
                        help="0=Active, 1=Applied",
                        width="medium"
                    ),
                    "Total": st.column_config.ProgressColumn(
                        "Engagement", 
                        format="%d", 
                        min_value=0, 
                        max_value=len(members_df)
                    )
                }
            )
            
            st.divider()
            st.markdown("#### 🔎 Detailed Vote Log")
            with st.expander("View Individual Votes"):
                if not votes_df.empty:
                    st.dataframe(votes_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No votes cast yet.")
        pass

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
            
        # Format: AAPL 150.20 ▲ +1.2%
        ticker_content += f"""
        <span style="margin-right: 30px; color: var(--text-color); font-weight: bold; font-family: 'Courier New', monospace;">
            {ticker} 
            <span style="color: {color}; margin-left:5px;">
                {price} {arrow} {pct:+.1f}%
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
    # TAB 1: CAPITAL INJECTION
    # ==========================================
    with t_invest:
        st.subheader("Request Capital Increase")
        st.caption("You may request to increase your stake during the first month of every quarter.")
        
        # Date Logic (from previous turn)
        today = datetime.now()
        open_months = [1, 4, 7, 10] 
        is_open = today.month in open_months
        
        # Status Check (Must be Active, liq_pending == 0)
        if user['liq_pending'] == 1:
            st.error("❌ Action Unavailable")
            st.write("You cannot add capital while a liquidation request is pending.")
        
        elif is_open:
            st.success(f"🟢 Window Open (Q{((today.month-1)//3)+1})")
            
            with st.form("top_up_form"):
                st.write("**How much would you like to add?**")
                st.caption("Limit: €1,000.00 per quarter.")
                
                amount = st.number_input("Amount (€)", min_value=50.0, max_value=1000.0, step=50.0)
                confirm = st.checkbox(f"I confirm I will transfer €{amount:.2f} to the TIC Treasury upon approval.")
                
                if st.form_submit_button("Submit Request"):
                    if confirm:
                        st.balloons()
                        st.success(f"Request for +€{amount:.2f} sent to the Treasurer. You will be contacted shortly.")
                    else:
                        st.warning("Please confirm the transfer agreement.")
        else:
            st.warning("🔒 Window Closed")
            st.markdown(f"Top-ups are only allowed in **Jan, Apr, Jul, and Oct**.")

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
                if current_p != user['p']:
                    st.error("❌ Incorrect current password.")
                elif new_p1 != new_p2:
                    st.error("❌ New passwords do not match.")
                elif len(new_p1) < 4:
                    st.warning("⚠️ Password is too short (min 4 chars).")
                else:
                    # Save to Google Sheet
                    if update_member_field_in_gsheet(user['u'], "Password", new_p1):
                        st.success("✅ Password updated successfully!")
                        st.info("Logging you out to re-authenticate...")
                        
                        # --- FORCE LOGOUT SEQUENCE ---
                        time.sleep(2)
                        st.session_state.clear()
                        st.rerun()
                    else:
                        st.error("❌ Update failed. Database error.")
            

def render_risk_macro_dashboard(f_port, q_port):
    st.title("⚠️ Risk & Macro Observatory")
    
    # 1. Define Tabs (Same as before)
    tabs = ["Correlation Matrix","Value at Risk (VaR)","Macro Indicators", "Market News"]
    
    # 2. State Logic (Use session state to preserve the tab choice)
    if 'risk_active_tab' not in st.session_state: st.session_state['risk_active_tab'] = tabs[0]
    try: curr_index = tabs.index(st.session_state['risk_active_tab'])
    except: curr_index = 0
    active_tab = st.radio("Risk View", tabs, index=curr_index, horizontal=True, label_visibility="collapsed")
    st.session_state['risk_active_tab'] = active_tab
    st.divider()

    # --- TAB 1: CORRELATION MATRIX ---
    if active_tab == "Correlation Matrix":
        st.subheader("Portfolio Correlation (Live)")
        
        # Radio to switch between views (Needs a unique key since it's a dedicated radio button)
        view_mode = st.radio("Select Portfolio View:", ["Fundamental Assets", "Quant Assets"], horizontal=True, key="corr_view_select")
        
        target_df = f_port if view_mode == "Fundamental Assets" else q_port
        
        if not target_df.empty:
            if 'ticker' in target_df.columns:
                col_name = 'ticker'
            elif 'model_id' in target_df.columns:
                col_name = 'model_id'
            else:
                col_name = None
                
            if col_name:
                my_tickers = [t for t in target_df[col_name].unique() if isinstance(t,str) and "CASH" not in t.upper()]
                
                if len(my_tickers) > 1:
                    with st.spinner(f"Calculating correlations for {len(my_tickers)} assets..."):
                        corr_matrix = fetch_correlation_data(my_tickers)
                    
                    if not corr_matrix.empty:
                        st.plotly_chart(
                            px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), 
                            use_container_width=True
                        )
                        
                        st.divider()
                        st.subheader("🧠 Risk Analysis")
                        high_risk = []
                        hedges = []
                        
                        cols = corr_matrix.columns
                        for i in range(len(cols)):
                            for j in range(i+1, len(cols)):
                                val = corr_matrix.iloc[i, j]
                                pair = f"{cols[i]} ↔ {cols[j]}"
                                if val > 0.85: high_risk.append(f"{pair} ({val:.2f})")
                                elif val < -0.5: hedges.append(f"{pair} ({val:.2f})")
                        
                        c_warn, c_info = st.columns(2)
                        with c_warn:
                            if high_risk: st.error(f"🚨 **Critical Concentration ({len(high_risk)} pairs)**")
                            else: st.success("✅ No critical concentration (>0.85).")
                            
                        with c_info:
                            if hedges: st.info(f"🛡️ **Natural Hedges ({len(hedges)} pairs)**")
                            else: st.write("No strong negative correlations.")

                else: st.info("Not enough assets to calculate correlation (Need 2+).")
            else: st.warning("Could not identify Ticker column.")
        else: st.warning("Portfolio is empty.")
    
    elif active_tab == "Value at Risk (VaR)":
        st.subheader("🛡️ Portfolio Historical VaR (95%)")
        st.caption("Estimating potential loss based on the last 1 year of historical data for current holdings.")
        
        # 1. Combine Portfolios for Calculation
        combined_tickers = []
        weights = {}
        total_aum = 0
        
        # Process Fundamental
        if not f_port.empty and 'ticker' in f_port.columns:
            for _, row in f_port.iterrows():
                t = str(row.get('ticker'))
                v = float(row.get('market_value', 0))
                if "CASH" not in t.upper() and v > 0:
                    combined_tickers.append(t)
                    weights[t] = weights.get(t, 0) + v
                    total_aum += v
                    
        # Process Quant
        if not q_port.empty:
            col = 'model_id' if 'model_id' in q_port.columns else 'ticker'
            for _, row in q_port.iterrows():
                t = str(row.get(col, ''))
                v = float(row.get('market_value', 0))
                if t and "CASH" not in t.upper() and v > 0:
                    combined_tickers.append(t)
                    weights[t] = weights.get(t, 0) + v
                    total_aum += v

        # 2. Fetch Data with Safety Checks
        if combined_tickers and total_aum > 0:
            unique_tickers = list(set(combined_tickers))
            with st.spinner("Calculating Historical VaR..."):
                try:
                    # Download history
                    data_raw = yf.download(unique_tickers, period="1y", progress=False)
                    
                    # Check if data is empty or corrupted
                    if data_raw.empty:
                        st.warning("⚠️ Could not fetch historical data. API might be rate-limited.")
                        return

                    # Handle single ticker vs multiple tickers structure
                    if len(unique_tickers) == 1:
                        data = data_raw['Close'].to_frame() if 'Close' in data_raw else pd.DataFrame()
                        data.columns = unique_tickers # Rename col to ticker
                    else:
                        data = data_raw['Close']
                        
                    if data.empty:
                        st.warning("⚠️ No 'Close' price data available.")
                        return

                    # Calculate Returns
                    returns = data.pct_change().dropna()
                    
                    if returns.empty:
                        st.warning("⚠️ Not enough historical data points to calculate VaR.")
                        return

                    # Calculate Portfolio Returns (Weighted)
                    # Filter weights to only include tickers we actually got data for
                    valid_tickers = [t for t in returns.columns if t in weights]
                    
                    if not valid_tickers:
                        st.warning("⚠️ No matching data found for your tickers.")
                        return

                    w_vector = [weights[t]/total_aum for t in valid_tickers]
                    
                    # Dot product: Matrix of returns * Weights
                    port_returns = returns[valid_tickers].dot(w_vector)
                    
                    # 3. Calculate Stats
                    confidence_level = 0.95
                    var_95 = np.percentile(port_returns, (1 - confidence_level) * 100)
                    cvar_95 = port_returns[port_returns <= var_95].mean()
                    
                    # Convert to currency
                    var_val = total_aum * var_95
                    cvar_val = total_aum * cvar_95
                    
                    # 4. Display Metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Daily VaR (95%)", f"€{var_val:,.2f}", f"{var_95*100:.2f}%")
                    c2.metric("Daily CVaR (Expected Tail Loss)", f"€{cvar_val:,.2f}", f"{cvar_95*100:.2f}%")
                    c3.metric("Calculation Base", f"{len(valid_tickers)} Assets", "1 Year History")
                    
                    # 5. Visualization (Histogram)
                    fig_hist = px.histogram(
                        port_returns, 
                        nbins=50, 
                        title="Distribution of Daily Portfolio Returns",
                        labels={'value': 'Daily Return'},
                        color_discrete_sequence=['#444444']
                    )
                    # Add VaR Line
                    fig_hist.add_vline(x=var_95, line_width=3, line_dash="dash", line_color="#FF4444", annotation_text="VaR 95%")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"VaR Calculation failed: {e}")
        else:
            st.warning("No active assets found in portfolio to calculate VaR.")
    
    # --- TAB 2: MACRO INDICATORS ---
    elif active_tab == "Macro Indicators":
        st.subheader("Global Markets")
        macro = fetch_macro_data()
        c1, c2, c3, c4 = st.columns(4)
        
        def show(col, lbl, k, fmt="%.2f"):
            d = macro.get(k, {}); curr = d.get('value', 0); delta = d.get('delta', 0)
            col.metric(lbl, fmt % curr, f"{delta:.2f}")
            
        show(c1, "🇺🇸 10Y Yield", '10Y Treasury', "%.2f%%")
        show(c2, "😨 VIX Index", 'VIX')
        show(c3, "💶 EUR/USD", 'EUR/USD', "%.4f")
        show(c4, "🛢️ Crude Oil", 'Crude Oil', "$%.2f")

    # --- TAB 3: MARKET NEWS ---
    elif active_tab == "Market News":
        st.subheader("Market Intelligence")
        news = fetch_macro_news()
        
        if not news:
            st.warning("RSS Feed unavailable or empty.")
        else:
            c_news1, c_news2 = st.columns(2)
            for i, item in enumerate(news):
                col = c_news1 if i % 2 == 0 else c_news2
                with col:
                    # FIX: Used markdown to render HTML with the link
                    st.markdown(f"""
                    <div class="news-item">
                        <div class="news-source">{item['source']} | {item['published']}</div>
                        <a class="news-head" href="{item['link']}" target="_blank">{item['title']}</a>
                        <div class="news-sum">{item['summary']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
def render_fundamental_dashboard(user, portfolio, proposals):
    st.title(f"📈 Fundamental Dashboard")
    
    # 1. RAW DATA EXPANDER
    with st.expander("📊 View Raw Portfolio Data"):
        st.dataframe(
            portfolio,
            use_container_width=True,
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
        fig.update_layout(yaxis_title="Rebased (100)", legend_title=None)
        
        # RENDER CHART INSIDE THE IF BLOCK
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough historical data to generate performance chart.")
    
    # 3. ALLOCATION & SECTOR ANALYSIS
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Allocation")
        if not portfolio.empty:
            fig_pie = px.pie(portfolio, values='target_weight', names='sector', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No assets to display.")

    with c2:
        st.subheader("Sector Performance")
        # --- REAL DATA LOGIC ---
        if not portfolio.empty and 'ticker' in portfolio.columns:
            # 1. Get tickers (excluding Cash)
            tickers = portfolio['ticker'].dropna().unique().tolist()
            
            # 2. Fetch Real Change % (cached)
            live_data = fetch_live_prices_with_change(tickers)
            
            # 3. Map the 'pct' change to the dataframe
            # Use a safe lambda to avoid errors if ticker missing
            portfolio['real_return'] = portfolio['ticker'].map(
                lambda t: live_data.get(str(t), {}).get('pct', 0.0)
            )
            
            # 4. Plot Treemap
            fig_tree = px.treemap(
                portfolio, 
                path=[px.Constant("Portfolio"), 'sector', 'ticker'], 
                values='target_weight', 
                color='real_return',
                color_continuous_scale='RdYlGn', # Red to Green
                color_continuous_midpoint=0,     # 0% is center
                hover_data=['real_return']
            )
            
            fig_tree.update_traces(hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<br>Change: %{customdata[0]:.2f}%')
            st.plotly_chart(fig_tree, use_container_width=True)

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
    
    st.divider()
    
    # 5. ACTIVE PROPOSALS
    st.header("🗳️ Active Proposals")
    current_props = [p for p in proposals if p.get('Dept') == 'Fundamental']
    
    if not current_props:
        st.info("No active proposals for Fundamental department.")

    for p in current_props:
        with st.container(border=True):
            c_a, c_b = st.columns([4, 1])
            c_a.subheader(f"{p.get('Type')}: {p.get('Item')}")
            c_a.write(p.get('Description'))
            c_a.caption(f"Closes: {p.get('End_Date')}")
            c_b.metric("Votes", "TBD")
            
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
                    use_container_width=True,
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
                st.plotly_chart(fig, use_container_width=True)

    # --- 3. DETAILED HOLDINGS ---
    st.divider()
    st.subheader("📜 Detailed Asset Holdings")
    
    if not portfolio.empty:
        display_cols = ['ticker', 'name', 'sector', 'model', 'units', 'market_value']
        valid_cols = [c for c in display_cols if c in portfolio.columns]
        
        st.dataframe(
            portfolio[valid_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "market_value": st.column_config.NumberColumn("Value", format="€%.2f"),
                "ticker": st.column_config.TextColumn("Asset"),
            }
        )
        
    st.divider()
    # MONTE CARLO SIMULATION
    st.markdown("### 🎲 Monte Carlo Risk Engine")
    st.caption("Project future portfolio performance based on random walk simulations (Geometric Brownian Motion).")
    
    c_param, c_plot = st.columns([1, 3])
    
    with c_param:
        st.write("**Settings**")
        n_sims = st.slider("Simulations", 10, 100, 50)
        horizon = st.slider("Horizon (Days)", 30, 365, 252)
        mu = st.slider("Expected Return", -10, 30, 8) / 100
        sigma = st.slider("Volatility", 5, 50, 15) / 100
        
    with c_plot:
        dt = 1/252
        S0 = 100
        dates = pd.date_range(start=datetime.now(), periods=horizon+1)
        sim_data = {'Date': dates}
        
        for i in range(n_sims):
            Z = np.random.normal(0, 1, horizon)
            daily_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            daily_returns = np.insert(daily_returns, 0, 1.0)
            sim_data[f'Sim_{i}'] = S0 * np.cumprod(daily_returns)
            
        df_mc = pd.DataFrame(sim_data).melt(id_vars='Date', var_name='Sim', value_name='Value')
        
        fig = px.line(df_mc, x='Date', y='Value', color='Sim', 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Value (Base 100)")
        fig.update_traces(opacity=0.5, line=dict(width=1))
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        final_vals = df_mc[df_mc['Date'] == dates[-1]]['Value']
        c1, c2, c3 = st.columns(3)
        c1.metric("Median", f"{final_vals.median():.0f}")
        c2.metric("Worst Case (5%)", f"{np.percentile(final_vals, 5):.0f}")
        c3.metric("Best Case (95%)", f"{np.percentile(final_vals, 95):.0f}")
        
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
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    
    # Unpack everything
    members, f_port, q_port, props, calendar_events, f_total, q_total, df_votes, nav_f, nav_q, att_df = load_data()

    if not st.session_state['logged_in']:
        c1, c2, c3 = st.columns([1,1.5,1])
        with c2:
            st.image(TIC_LOGO, width=200)
            st.title("TIC Portal")
            st.info("Welcome to the Internal Management System")

            st.caption("By logging in, you acknowledge that this portal is for informational purposes only and does not constitute a binding financial statement.")
            
            # --- NEW: CLEAN, ALWAYS-VISIBLE LOGIN FORM ---
            with st.form("login_form", clear_on_submit=True):
                st.subheader("Member Login")
                
                u = st.text_input("Username", key="login_u")
                p = st.text_input("Password", type="password", key="login_p")
                
                # Create two columns for the buttons
                c_log, c_guest = st.columns(2)
                
                # --- BUTTON 1: MEMBER LOGIN ---
                if c_log.form_submit_button("Log In", type="primary"):
                    user = authenticate(u, p, members)
                    if user is not None:
                        st.session_state['user'] = user.to_dict()
                        st.session_state['logged_in'] = True
                        
                        # 1. Track Last Login (Google Sheets)
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                        update_member_field_in_gsheet(user['u'], "Last Login", timestamp)
                        
                        # 2. Time-aware greeting
                        h = datetime.now().hour
                        if 5 <= h < 12: greeting = "Good Morning"
                        elif 12 <= h < 18: greeting = "Good Afternoon"
                        else: greeting = "Good Evening"
                        st.toast(f"🚀 {greeting}, {user['n']}!", icon="👋")
                        
                        st.rerun()
                    else: 
                        st.error("Invalid Username or Password")
                
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
            submit = st.form_submit_button("GO", use_container_width=True)
            
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
        
        # 3. Render the Navigation
        nav = st.radio("Navigation", menu, index=default_index)
        
        # 4. Background Save (Optimized)
        clean_nav = nav.split(" (")[0]
        
        # Only trigger updates if the page actually changed
        if clean_nav != st.session_state.get('previous_choice', ''):
            st.session_state['previous_choice'] = clean_nav
            
            # Update local session immediately so it feels instant
            if user.get('last_page') != clean_nav:
                st.session_state['user']['last_page'] = clean_nav
                
                # THREADING: Send API call to background thread to prevent UI lag
                if user['r'] != 'Guest':
                    threading.Thread(
                        target=update_member_field_in_gsheet, 
                        args=(user['u'], "Last_Page", clean_nav)
                    ).start()
        st.divider()
        if st.button("Log Out"): st.session_state.clear(); st.rerun()
        
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
        # 1. BOARD, ADVISORY, AND GUESTS (See Both)
        if user['d'] in ['Board', 'Advisory'] or user['r'] == 'Guest':
            st.title("🏛️ Executive Overview (Guest View)" if user['r'] == 'Guest' else "🏛️ Executive Overview")
            
            if user['r'] == 'Guest':
                st.info("👀 You are viewing the live portfolio in Read-Only mode.")
            
            t_fund, t_quant = st.tabs(["📈 Fundamental", "🤖 Quant"])
            
            with t_fund: 
                render_fundamental_dashboard(user, f_port, props)
                st.divider()
                render_voting_section(user, props, df_votes, "Fundamental")
                
            with t_quant: 
                render_quant_dashboard(user, q_port, props)
                st.divider()
                render_voting_section(user, props, df_votes, "Quant")

        # 2. QUANT TEAM (See Quant Only)
        elif user['d'] == 'Quant': 
            render_quant_dashboard(user, q_port, props)
            st.divider()
            render_voting_section(user, props, df_votes, "Quant")
            
        # 3. FUNDAMENTAL / GENERAL (See Fundamental Only)
        else: 
            render_fundamental_dashboard(user, f_port, props)
            st.divider()
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













































































































