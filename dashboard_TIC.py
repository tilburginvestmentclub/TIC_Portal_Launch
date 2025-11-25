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

def get_data_from_sheet(worksheet_name):
    """Helper to fetch all records from a specific tab."""
    client = init_connection()
    if not client: return pd.DataFrame()
    
    try:
        # Open the Master Sheet
        sheet = client.open("TIC_Database_Master")
        # Open the specific tab (Worksheet)
        worksheet = sheet.worksheet(worksheet_name)
        # Get all values
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error reading tab '{worksheet_name}': {e}")
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

@st.cache_data(ttl=3600)
def fetch_correlation_data(tickers):
    # Filter out empty or non-string values just in case
    valid_tickers = [str(t) for t in tickers if str(t) != 'nan' and t]
    
    if len(valid_tickers) < 2:
        return pd.DataFrame() # Need at least 2 stocks for correlation
        
    try:
        # Download data for these specific tickers
        df = yf.download(valid_tickers, period="1y", progress=False)['Close']
        if df.empty: return pd.DataFrame()
        return df.corr()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_live_prices_with_change(tickers):
    """Fetches Price AND Hourly Change for all tickers."""
    if not tickers: return {}
    
    # Clean tickers
    clean_tickers = list(set([
        str(t) for t in tickers 
        if isinstance(t, str) and t and "CASH" not in t.upper()
    ]))
    
    try:
        # Fetch 1 day of data with 1-hour intervals to get the "Last Hour" change
        data = yf.download(
            clean_tickers, 
            period="1d", 
            interval="1h", 
            group_by='ticker', 
            progress=False,
            threads=True
        )
        
        results = {}
        
        # Handle Single Ticker vs Multiple Ticker structure in yfinance
        is_multi = len(clean_tickers) > 1
        
        for t in clean_tickers:
            try:
                # Get the specific dataframe for this ticker
                df = data[t] if is_multi else data
                
                # We need at least 2 rows to calculate a change
                valid_rows = df['Close'].dropna()
                
                if len(valid_rows) >= 1:
                    current_price = valid_rows.iloc[-1]
                    
                    # Calculate change from previous hour (if available)
                    if len(valid_rows) >= 2:
                        prev_price = valid_rows.iloc[-2]
                        change = current_price - prev_price
                        pct_change = (change / prev_price) * 100
                    else:
                        change = 0.0
                        pct_change = 0.0
                        
                    results[t] = {
                        'price': round(float(current_price), 2),
                        'change': round(float(change), 2),
                        'pct': round(float(pct_change), 2)
                    }
                else:
                    results[t] = {'price': 0.0, 'change': 0.0, 'pct': 0.0}
                    
            except Exception:
                results[t] = {'price': 0.0, 'change': 0.0, 'pct': 0.0}
                
        return results
        
    except Exception as e:
        print(f"Ticker Error: {e}")
        return {}
    
@st.cache_data(ttl=3600*4) # Cache for 4 hours
@st.cache_data(ttl=3600*4)
def fetch_real_benchmark_data(portfolio_df):
    """
    Calculates a Weighted Return Index (Growth of 100) to avoid currency mixing issues.
    """
    end = datetime.now()
    start = end - timedelta(days=180)
    
    # 1. Setup the DataFrame with Dates
    # We assume business days (B) to align markets roughly
    dates = pd.date_range(start=start, end=end, freq='B')
    df_chart = pd.DataFrame(index=dates)
    
    # 2. Fetch Benchmark (S&P 500)
    try:
        sp500 = yf.download('^GSPC', start=start, end=end, progress=False)['Close']
        # Normalize to 100 (Growth of $100)
        if not sp500.empty:
            df_chart['SP500'] = (sp500 / sp500.iloc[0]) * 100
    except:
        df_chart['SP500'] = 100.0

    # 3. Fetch TIC Portfolio
    if not portfolio_df.empty and 'ticker' in portfolio_df.columns:
        try:
            # Filter Valid Tickers (No Cash)
            tickers = [
                t for t in portfolio_df['ticker'].unique() 
                if isinstance(t, str) and "CASH" not in t.upper()
            ]
            
            # Get Cash Weight
            # If you have rows like 'CASH_EUR', sum their weights
            cash_weight = 0.0
            if 'target_weight' in portfolio_df.columns:
                cash_rows = portfolio_df[portfolio_df['ticker'].str.contains("CASH", na=False)]
                cash_weight = pd.to_numeric(cash_rows['target_weight'], errors='coerce').sum()

            if tickers:
                # Download Equity Data
                data = yf.download(tickers, start=start, end=end, progress=False)['Close']
                data = data.ffill().bfill() # Fill gaps (holidays)
                
                # Reindex to match our main timeline
                data = data.reindex(df_chart.index, method='ffill')
                
                # Calculate Individual Stock Indexes (Start = 100)
                # This creates a "Growth" curve for every single stock in local currency
                # (10% growth in USD is the same shape as 10% growth in EUR)
                stock_indexes = (data / data.iloc[0]) * 100
                
                # Apply Weights
                weighted_returns = pd.Series(0.0, index=df_chart.index)
                
                for t in tickers:
                    if t in stock_indexes.columns:
                        # Get weight from your Google Sheet
                        w = portfolio_df.loc[portfolio_df['ticker'] == t, 'target_weight'].iloc[0]
                        # Add weighted contribution to the index
                        weighted_returns += stock_indexes[t] * float(w)
                
                # Add Cash Contribution (Cash stays at 100, it doesn't grow/shrink in this view)
                # If you have 20% cash, that portion of the portfolio stays flat
                weighted_returns += (100.0 * cash_weight)
                
                df_chart['TIC_Fund'] = weighted_returns
            else:
                df_chart['TIC_Fund'] = 100.0
                
        except Exception as e:
            print(f"Bench Error: {e}")
            df_chart['TIC_Fund'] = 100.0
    else:
        df_chart['TIC_Fund'] = 100.0

    # Clean up for plotting
    df_chart = df_chart.dropna().reset_index().rename(columns={'index':'Date'})
    return df_chart
    
    @st.cache_data(ttl=60)
    def load_data():
    # --- 1. CONFIGURATION ---
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
        'hb': { 'r': "Head of Business Development", 'd': 'Board', 's': 'Business Dev', 'admin': False},
        'other': {'r': 'Member', 'd': 'General', 's': 'General', 'admin': False},
    }

    def clean_float(val):
        if pd.isna(val) or val == '': return 0.0
        try: return float(str(val).replace('€', '').replace(',', '').replace(' ', ''))
        except: return 0.0

    # --- 2. LOAD PORTFOLIOS & CALCULATE LIVE ASSETS ---
    f_port = get_data_from_sheet("Fundamentals")
    q_port = get_data_from_sheet("Quant")
    
    # HELPER: Calculate Real-Time Fund Value (Robust Version)
    def calculate_live_total(df):
        total_val = 0.0
        
        if not df.empty:
            # Normalize headers
            df.columns = df.columns.astype(str).str.lower().str.strip()
            
            # 1. Identify the Ticker Column
            ticker_col = None
            if 'ticker' in df.columns: ticker_col = 'ticker'
            elif 'model_id' in df.columns: ticker_col = 'model_id'
            
            # 2. If no ticker column found, fallback to sheet total
            if not ticker_col:
                if 'total' in df.columns:
                    return clean_float(df['total'].iloc[0]), df
                return 0.0, df

            # 3. Get Tickers (No Cash)
            tickers = [t for t in df[ticker_col].unique() if isinstance(t, str) and "CASH" not in t.upper()]
            
            # 4. Fetch Live Prices
            prices = fetch_live_prices_with_change(tickers)
            
            # 5. Sum up (Equity * Price) + Cash
            for index, row in df.iterrows():
                ticker = str(row.get(ticker_col, ''))
                # Handle varied column names for 'Units'
                units = 0.0
                if 'units' in df.columns: units = clean_float(row.get('units', 0))
                elif 'allocation' in df.columns: units = clean_float(row.get('allocation', 0)) # Fallback if Quant uses allocation as units
                
                # Handle varied column names for 'Market Value'
                sheet_val = 0.0
                if 'market_value' in df.columns: sheet_val = clean_float(row.get('market_value', 0))
                elif 'total' in df.columns: sheet_val = clean_float(row.get('total', 0)) # Fallback
                
                if "CASH" in ticker.upper():
                    val = sheet_val
                else:
                    live_price = prices.get(ticker, {}).get('price', 0.0)
                    if live_price > 0 and units > 0:
                        val = live_price * units
                    else:
                        val = sheet_val # Fallback
                
                total_val += val
                
        return total_val, df

    # Calculate Live Totals
    f_total, f_port = calculate_live_total(f_port)
    q_total, q_port = calculate_live_total(q_port)

    # Standardize Quant DataFrame for Charts (Ensure model_id exists)
    if not q_port.empty:
        if 'ticker' in q_port.columns: 
            q_port = q_port.rename(columns={'ticker': 'model_id'})
        if 'target_weight' in q_port.columns: 
            q_port = q_port.rename(columns={'target_weight': 'allocation'})
        
        # Ensure allocation is numeric
        if 'allocation' in q_port.columns:
            q_port['allocation'] = q_port['allocation'].apply(clean_float)

    # --- 3. LOAD MEMBERS & CALCULATE NAV ---
    df_mem = get_data_from_sheet("Members")
    members_list = []
    
    total_units_fund = 0.0
    total_units_quant = 0.0
    
    if not df_mem.empty:
        # Clean headers
        df_mem.columns = df_mem.columns.astype(str).str.strip()
        
        if 'Units_Fund' in df_mem.columns:
            total_units_fund = pd.to_numeric(df_mem['Units_Fund'], errors='coerce').fillna(0).sum()
        if 'Units_Quant' in df_mem.columns:
            total_units_quant = pd.to_numeric(df_mem['Units_Quant'], errors='coerce').fillna(0).sum()

    nav_fund = f_total / total_units_fund if total_units_fund > 0 else 100.00
    nav_quant = q_total / total_units_quant if total_units_quant > 0 else 100.00

    if not df_mem.empty:
        for _, row in df_mem.iterrows():
            role_code = str(row.get('Role', 'other')).strip().lower()
            role_data = ROLE_MAP.get(role_code, {'r': 'Member', 'd': 'General', 's': 'General', 'admin': False})
            name = str(row.get('Name', 'Unknown')).strip()
            uname = name.lower().replace(" ", ".")
            email = str(row.get('Email', f"{uname}@tilburg.edu")).strip()
            
            try: liq_val = int(float(row.get('Liq Pending', 0)))
            except: liq_val = 0
            
            u_f = clean_float(row.get('Units_Fund', 0))
            u_q = clean_float(row.get('Units_Quant', 0))
            real_value = (u_f * nav_fund) + (u_q * nav_quant)
            
            members_list.append({
                'u': uname,
                'p': str(row.get('Password', 'pass')).strip(),
                'n': name,
                'email': email,
                'r': role_data['r'],
                'd': role_data['d'],
                's': role_data['s'],
                'admin': role_data.get('admin', False),
                'status': 'Pending' if liq_val == 1 else 'Active',
                'liq_pending': liq_val,
                'contribution': clean_float(row.get('Initial Investment', 0)),
                'value': real_value,
                'units_fund': u_f,
                'units_quant': u_q,
                'contract_text': "TIC MEMBERSHIP..."
            })
    else:
        members_list = [{'u': 'admin', 'p': 'pass', 'n': 'Offline', 'r': 'Admin', 'd': 'Board', 'admin': True, 'value': 0}]
    
    members = pd.DataFrame(members_list)

    # --- 4. MESSAGES ---
    msgs = get_data_from_sheet("Messages")
    if not msgs.empty: msgs.columns = msgs.columns.str.lower()
    messages = msgs.to_dict('records') if not msgs.empty else []
    
    # --- 5. EVENTS ---
    evts = get_data_from_sheet("Events")
    manual_events = []
    if not evts.empty:
        evts['Date'] = pd.to_datetime(evts['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        for _, row in evts.iterrows():
            manual_events.append({
                'title': str(row.get('Title','')),
                'ticker': str(row.get('Ticker','')),
                'date': str(row.get('Date','')),
                'type': str(row.get('Type','')).lower(),
                'audience': str(row.get('Audience','all'))
            })

    real_events = []
    if not f_port.empty and 'ticker' in f_port.columns:
        tickers = [t for t in f_port['ticker'].dropna().unique() if isinstance(t,str) and "CASH" not in t]
        real_events = fetch_company_events(tickers)
    full_calendar = real_events + manual_events

    # --- 6. VOTING ---
    df_props = get_data_from_sheet("Proposals")
    proposals = []
    if not df_props.empty:
        df_props['ID'] = df_props['ID'].astype(str)
        proposals = df_props.to_dict('records')

    df_votes = get_data_from_sheet("Votes")
    if not df_votes.empty:
        df_votes['Proposal_ID'] = df_votes['Proposal_ID'].astype(str)

    return members, f_port, q_port, messages, proposals, full_calendar, f_total, q_total, df_votes, nav_fund, nav_quant
    
# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

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
        
def send_new_message_gsheet(from_user, to_user, subject, body):
    """Appends a new message row to the 'Messages' tab."""
    client = init_connection()
    if not client: return False
    
    try:
        sheet = client.open("TIC_Database_Master")
        ws = sheet.worksheet("Messages")
        
        new_row = [
            int(datetime.now().timestamp()), # ID
            datetime.now().strftime("%Y-%m-%d %H:%M"), # Timestamp
            from_user,
            to_user,
            subject,
            body,
            "False" # Read
        ]
        
        ws.append_row(new_row)
        return True
    except Exception as e:
        st.error(f"Message send failed: {e}")
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

def create_pdf_report(fund_port, quant_port):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="TIC Quarterly Report", ln=1, align='C')
    return pdf.output(dest='S').encode('latin-1')

def send_new_message(from_user, to_user, subject, body):
    file_path = "data/TIC_Messages.xlsx"
    
    new_data = {
        'ID': int(datetime.now().timestamp()), # Simple unique ID
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'From_User': from_user,
        'To_User': to_user,
        'Subject': subject,
        'Body': body,
        'Read': False
    }
    
    if os.path.exists(file_path):
        df = pd.read_excel(file_path, engine='openpyxl')
        # Append new row
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_excel(file_path, index=False)
        return True
    return False

# ==========================================
# 4. VIEW COMPONENTS
# ==========================================
def render_voting_section(user, proposals, votes_df, target_dept):
    """Renders the voting UI for a specific department."""
    st.header(f"🗳️ {target_dept} Governance")
    
    # Filter proposals for this Dept (Active only)
    # We check 'Dept' matches AND 'Applied' is 0 (Active)
    active_props = [p for p in proposals if p.get('Dept') == target_dept and str(p.get('Applied')) == '0']
    
    if not active_props:
        st.info("No active proposals.")
        return

    for p in active_props:
        p_id = str(p['ID'])
        
        with st.container(border=True):
            c_desc, c_act = st.columns([3, 1])
            
            with c_desc:
                st.subheader(f"{p['Type']}: {p['Item']}")
                st.write(p['Description'])
                st.caption(f"Ends: {p['End_Date']}")
                
                # Calculate Results
                if not votes_df.empty:
                    relevant_votes = votes_df[votes_df['Proposal_ID'] == p_id]
                    yes_count = len(relevant_votes[relevant_votes['Vote'] == 'YES'])
                    no_count = len(relevant_votes[relevant_votes['Vote'] == 'NO'])
                else:
                    yes_count, no_count = 0, 0
                
                # Progress Bar
                total = yes_count + no_count
                if total > 0:
                    yes_pct = yes_count / total
                    st.progress(yes_pct, text=f"Yes: {yes_count} | No: {no_count}")
                else:
                    st.write("No votes yet.")

            with c_act:
                # 1. Check if User has already voted
                user_has_voted = False
                if not votes_df.empty:
                    user_vote = votes_df[
                        (votes_df['Proposal_ID'] == p_id) & 
                        (votes_df['Username'] == user['u'])
                    ]
                    if not user_vote.empty:
                        user_has_voted = True
                
                # 2. Render Buttons
                if user_has_voted:
                    st.success("✅ Voted")
                else:
                    c_y, c_n = st.columns(2)
                    if c_y.button("YES", key=f"y_{p_id}"):
                        cast_vote_gsheet(p_id, user['u'], "YES")
                        st.rerun()
                    if c_n.button("NO", key=f"n_{p_id}"):
                        cast_vote_gsheet(p_id, user['u'], "NO")
                        st.rerun()
                
                # 3. Admin "Apply" Button
                # Only show if Admin AND vote is passing (simple majority)
                if user.get('admin', False) and total > 0 and yes_count > no_count:
                    st.divider()
                    if st.button("Execute & Close", key=f"exe_{p_id}"):
                        if mark_proposal_applied(p_id):
                            st.success("Proposal Applied!")
                            st.rerun()

def render_leaderboard(current_user):
    st.title("🏆 Trading Leaderboard")
    
    # Mock Leaderboard Data
    data = [
        {'Rank': 1, 'Member': 'Alvise (Quant)', 'Return': 18.4, 'Equity': 118400},
        {'Rank': 2, 'Member': 'Senyo (Board)', 'Return': 14.2, 'Equity': 114200},
        {'Rank': 3, 'Member': 'Boaz (Alumni)', 'Return': 9.1, 'Equity': 109100},
        {'Rank': 4, 'Member': 'Chris (Fund)', 'Return': 5.5, 'Equity': 105500},
    ]
    
    # Calculate Current User stats
    cash = st.session_state.get('shadow_cash', 100000)
    holdings = st.session_state.get('shadow_holdings', {})
    # Simple price fetch mock
    prices = {k: 150 for k in holdings.keys()} 
    equity = sum([v * prices[k] for k, v in holdings.items()])
    total_val = cash + equity
    user_return = ((total_val - 100000) / 100000) * 100
    
    user_entry = {
        'Rank': 0, # Placeholder
        'Member': f"{current_user['n']} (You)", 
        'Return': user_return, 
        'Equity': total_val
    }
    
    data.append(user_entry)
    df = pd.DataFrame(data).sort_values(by='Return', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    
    # Highlight user row
    def highlight_user(row):
        return ['background-color: #262730; font-weight: bold' if '(You)' in row['Member'] else '' for _ in row]

    st.dataframe(
        df.style.apply(highlight_user, axis=1).format({'Return': "{:.2f}%", 'Equity': "€{:,.0f}"}),
        use_container_width=True,
        hide_index=True
    )

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

    # Calculation Logic
    future_fcf = []
    discount_factors = []
    
    # Years 1-5
    for i in range(1, 6):
        val = fcf * ((1 + growth_1_5) ** i)
        future_fcf.append(val)
        discount_factors.append((1 + wacc) ** i)
        
    # Years 6-10
    for i in range(1, 6):
        val = future_fcf[-1] * ((1 + growth_6_10) ** i)
        future_fcf.append(val)
        discount_factors.append((1 + wacc) ** (5 + i))
        
    # Terminal Value
    tv = (future_fcf[-1] * (1 + term_growth)) / (wacc - term_growth)
    pv_tv = tv / ((1 + wacc) ** 10)
    
    pv_fcf = sum([f/d for f, d in zip(future_fcf, discount_factors)])
    enterprise_value = pv_fcf + pv_tv
    equity_value = enterprise_value + cash
    share_price = equity_value / shares

    with c_viz:
        st.subheader(f"IMPLIED VALUE: ${share_price:.2f}")
        
        # Waterfall Chart - Styled for Dark Mode
        fig = px.bar(
            x=['PV FCF (10yr)', 'PV Term Val', 'Net Cash'],
            y=[pv_fcf, pv_tv, cash],
            title=f"ENTERPRISE VALUE BRIDGE: {ticker.upper()}",
            labels={'y':'Value ($B)', 'x':''}
        )
        # Bloomberg Orange, Dark Gray, Green
        fig.update_traces(marker_color=['#FF9900', '#444444', '#00FF00']) 
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0', family="Courier New"),
            title_font=dict(size=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.markdown("#### SENSITIVITY CHECK")
        st.write(f"At **{(wacc*100):.1f}% WACC** and **{(term_growth*100):.1f}% Terminal Growth**, the fair value is **${share_price:.2f}**.")
        st.latex(r"IV = \frac{\sum PV(FCF) + PV(TV) + Cash}{Shares}")

def render_calendar_view(user, all_events):
    st.title("🗓️ Smart Calendar")
    st.caption(f"Showing events for: {user['n']} ({user['d']} Dept)")
    
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
            # FIXED: Capture these boolean values to use in the filter logic below
            show_market = st.checkbox("Market Earnings", value=True)
            show_macro = st.checkbox("Macro Data", value=True)
            show_meet = st.checkbox("Meetings", value=True)
            
            # Helper to check permissions
            def can_view(event_audience):
                if event_audience == 'all': return True
                if event_audience == user['d']: return True # Matches Department
                if user['d'] == 'Board' and event_audience != 'all': 
                    return event_audience == 'Board'
                return False

            # 1. Filter by Permissions
            my_events = [e for e in all_events if can_view(e['audience'])]
            
            count = len([e for e in my_events if datetime.strptime(e['date'], '%Y-%m-%d').month == month])
            st.info(f"You have {count} relevant events this month.")

    with col_cal:
        # Create Calendar Grid
        cal = calendar.monthcalendar(year, month)
        cols = st.columns(7)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for i, day in enumerate(days):
            cols[i].markdown(f"**{day}**", unsafe_allow_html=True)
        
        for week in cal:
            cols = st.columns(7)
            for i, day in enumerate(week):
                if day == 0:
                    cols[i].write("")
                    continue
                
                day_str = f"{year}-{month:02d}-{day:02d}"
                
                # Get events for this specific day
                daily_events = [e for e in my_events if e['date'] == day_str]
                
                # Visual Styling
                is_today = (day == datetime.now().day and month == datetime.now().month)
                border = "2px solid #D4AF37" if is_today else "1px solid rgba(49, 51, 63, 0.2)"
                bg = "rgba(49, 51, 63, 0.1)"
                
                html_events = ""
                for e in daily_events:
                    # FIXED: Apply the Toggle Filters here
                    if e['type'] == 'market' and not show_market: continue
                    if e['type'] == 'macro' and not show_macro: continue
                    if e['type'] == 'meeting' and not show_meet: continue

                    # Color Logic
                    color = "#0068c9" # Blue (Market default)
                    if e['type'] == 'meeting':
                        if e['audience'] == 'all': color = "#228B22" # Green
                        else: color = "#800080" # Purple (Restricted)
                    
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

def render_admin_panel(user, members_df, f_port, q_port, total_aum, proposals, votes_df, nav_f, nav_q):
    st.title("🔒 Admin Console")
    st.info(f"Logged in as: {user['n']} ({user['r']})")
    
    # Define Path for saving member list
    MEMBER_FILE_PATH = "data/Member List.xlsx"
    
    # 1. TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 Member Database", "💰 Treasury", "📄 Reporting", "✅ Attendance", "🗳️ Governance"])
    
    # --- TAB 1: MEMBER DATABASE (READ/WRITE) ---
    with tab1:
        st.subheader("Manage Membership")
        st.markdown("Edit roles, emails, or status directly below.")
        if 'members_db' not in st.session_state: st.session_state['members_db'] = members_df
        
        edited_df = st.data_editor(
            st.session_state['members_db'][['n', 'email', 'r', 'd', 's', 'contribution', 'value', 'status']],
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
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

    # --- TAB 2: TREASURY & LIQUIDATION ---
    with tab2:
        with st.expander("💰 Process Capital Injection (Treasurer)"):
            c_nav1, c_nav2 = st.columns(2)
            c_nav1.metric("NAV Fundamental", f"€{nav_f:.2f}")
            c_nav2.metric("NAV Quant", f"€{nav_q:.2f}")
            
            with st.form("issue_units"):
                target_user = st.selectbox("Member", members_df['u'].tolist())
                amount = st.number_input("Cash Injected (€)", min_value=0.0)
                
                # Treasurer chooses where the money goes
                fund_choice = st.radio("Allocate to:", ["Fundamental", "Quant", "50/50 Split"])
                
                if st.form_submit_button("Issue Units"):
                    client = init_connection()
                    sheet = client.open("TIC_Database_Master")
                    ws = sheet.worksheet("Members")
                    
                    # Find user row
                    data = ws.get_all_records()
                    df = pd.DataFrame(data)
                    df['u'] = df['Name'].astype(str).str.lower().str.strip().str.replace(' ', '.')
                    idx = df.index[df['u'] == target_user].tolist()[0]
                    row_num = idx + 2
                    
                    # Calculate Units
                    u_f_add = 0.0
                    u_q_add = 0.0
                    
                    if fund_choice == "Fundamental":
                        u_f_add = amount / nav_f
                    elif fund_choice == "Quant":
                        u_q_add = amount / nav_q
                    else: # 50/50
                        u_f_add = (amount * 0.5) / nav_f
                        u_q_add = (amount * 0.5) / nav_q
                    
                    # Update Cells
                    curr_f = float(df.iloc[idx]['Units_Fund']) if 'Units_Fund' in df.columns else 0.0
                    curr_q = float(df.iloc[idx]['Units_Quant']) if 'Units_Quant' in df.columns else 0.0
                    curr_inv = float(df.iloc[idx]['Initial Investment'])
                    
                    # We find column indices dynamically
                    headers = ws.row_values(1)
                    
                    ws.update_cell(row_num, headers.index('Units_Fund')+1, curr_f + u_f_add)
                    ws.update_cell(row_num, headers.index('Units_Quant')+1, curr_q + u_q_add)
                    ws.update_cell(row_num, headers.index('Initial Investment')+1, curr_inv + amount)
                    
                    st.success(f"Issued: {u_f_add:.2f} Fund Units and {u_q_add:.2f} Quant Units.")
                    st.cache_data.clear()
        st.subheader("Club Financials")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cash Balance", "€12,450.00", "+€500")
        c2.metric("Pending Payouts", "€0.00")
        # DYNAMIC AUM: Uses the passed argument
        c3.metric("Total AUM", f"€{total_aum:,.2f}")
        
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

    # --- TAB 3: REPORTING ---
    with tab3:
        st.subheader("Export Data")
        if st.button("Generate PDF"):
            b64 = base64.b64encode(create_pdf_report(f_port, q_port)).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="TIC_Report.pdf">Download PDF</a>', unsafe_allow_html=True)
    
    # --- TAB 4: ATTENDANCE ---
    with tab4:
        st.subheader("Meeting Attendance Tracker")
        dates = ['2023-10-01', '2023-10-15', '2023-11-01', '2023-11-15']
        num_members = len(members_df)
        
        if 'attendance_df' not in st.session_state:
            att_data = {
                'Member': members_df['n'].tolist(),
            }
            for d in dates:
                att_data[d] = np.random.choice([True, False], size=num_members, p=[0.8, 0.2])
            st.session_state['attendance_df'] = pd.DataFrame(att_data)
        
        edited_att = st.data_editor(
            st.session_state['attendance_df'],
            use_container_width=True,
            column_config={d: st.column_config.CheckboxColumn(d) for d in dates},
            hide_index=True
        )
        
        if st.button("Save Attendance Log"):
            st.session_state['attendance_df'] = edited_att
            st.success("Attendance records updated.")
        st.metric("Average Attendance Rate", "85%")
        
    # --- TAB 5: GOVERNANCE ---
    with tab5:
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

def render_ticker_tape(data_dict):
    """Renders a ticker tape using the delta from the last hour."""
    
    # 1. Market Status Indicator
    now = datetime.now()
    is_open = 15 <= now.hour <= 22 # Approx US Market Hours in CET
    status_dot = "🟢" if is_open else "🔴"
    status_text = "MARKET OPEN 🟢" if is_open else "MARKET CLOSED 🔴"
    
    # 2. Build Ticker HTML
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
    
    # 3. Add Status at the start
    full_content = f'<span style="margin-right: 50px; color: #D4AF37; font-weight: 900; letter-spacing: 1px;">{status_dot} {status_text}</span>' + ticker_content

    # 4. Render Animation
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
    st.title("⚙️ Settings & Financials")
    
    t_invest, t_status = st.tabs(["🟢 Capital Injection", "🔴 Exit / Liquidation"])
    
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
            
def render_simulation(user):
    st.title("🎮 Paper Trading Simulation")
    if 'shadow_cash' not in st.session_state: st.session_state['shadow_cash'] = 100000.0
    if 'shadow_holdings' not in st.session_state: st.session_state['shadow_holdings'] = {}

    cash = st.session_state['shadow_cash']
    holdings = st.session_state['shadow_holdings']
    prices = fetch_live_prices(list(holdings.keys())) if holdings else {}
    equity = sum([u * prices.get(t,0) for t,u in holdings.items()])
    
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio Value", f"€{cash+equity:,.2f}")
        c2.metric("Cash Available", f"€{cash:,.2f}")
        c3.metric("Equity Value", f"€{equity:,.2f}")

    st.markdown("### Trading Desk")
    c_trade, c_hold = st.columns([1, 2])
    
    with c_trade:
        with st.container(border=True):
            st.subheader("New Order")
            with st.form("trade"):
                t = st.text_input("Ticker (e.g. TSLA)").upper()
                act = st.selectbox("Side", ["BUY", "SELL"])
                u = st.number_input("Units", 1)
                if st.form_submit_button("Execute Order", type="primary"):
                    p = fetch_live_prices([t]).get(t, 100)
                    cost = p * u
                    if act == "BUY":
                        if cash >= cost:
                            st.session_state['shadow_cash'] -= cost
                            st.session_state['shadow_holdings'][t] = holdings.get(t, 0) + u
                            st.success(f"Bought {u} @ {p}")
                            st.rerun()
                        else: st.error("Insufficient Cash")
                    elif act == "SELL":
                        if holdings.get(t, 0) >= u:
                            st.session_state['shadow_cash'] += cost
                            st.session_state['shadow_holdings'][t] -= u
                            if st.session_state['shadow_holdings'][t] == 0: del st.session_state['shadow_holdings'][t]
                            st.success(f"Sold {u} @ {p}")
                            st.rerun()
                        else: st.error("Insufficient Position")

    with c_hold:
        with st.container(border=True):
            st.subheader("Current Positions")
            if holdings:
                df = pd.DataFrame([{'Ticker':t, 'Units':u, 'Price':prices.get(t,0), 'Value':u*prices.get(t,0)} for t,u in holdings.items()])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else: st.caption("Your portfolio is empty. Start trading.")

def render_risk_macro_dashboard(portfolio):
    st.title("⚠️ Risk & Macro Observatory")
    
    news = fetch_macro_news()
    
    t1, t2, t3 = st.tabs(["Correlation Matrix", "Macro Indicators", "Market Intelligence"])
    
    with t1:
        st.subheader("Portfolio Correlation (Live)")
        
        # 1. Extract tickers from the portfolio dataframe
        if not portfolio.empty and 'ticker' in portfolio.columns:
            # Get unique tickers, drop N/A
            my_tickers = portfolio['ticker'].dropna().unique().tolist()
            
            if len(my_tickers) > 1:
                with st.spinner(f"Calculating correlations for {len(my_tickers)} assets..."):
                    corr_matrix = fetch_correlation_data(my_tickers)
                
                if not corr_matrix.empty:
                    st.plotly_chart(px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
                else:
                    st.warning("Could not fetch market data. Please check ticker symbols in Excel.")
            else:
                st.info("Add at least 2 distinct assets to 'Fundamentals' to see the correlation matrix.")
        else:
            st.warning("Portfolio is empty. Add positions in TIC_Portfolios.xlsx")
    
    with t2:
        st.subheader("Global Markets")
        macro = fetch_macro_data()
        c1, c2, c3, c4 = st.columns(4)
        def show(col, lbl, k, fmt="%.2f"):
            d = macro.get(k, {})
            col.metric(lbl, fmt % d.get('value',0), f"{d.get('delta',0):.2f}")
        show(c1, "10Y Yield", '10Y Treasury', "%.2f%%")
        show(c2, "VIX", 'VIX')
        show(c3, "EUR/USD", 'EUR/USD', "%.4f")
        show(c4, "Oil", 'Crude Oil', "$%.2f")
        
    with t3:
        if not news: st.warning("RSS Feed unavailable.")
        else:
            c_news1, c_news2 = st.columns(2)
            for i, item in enumerate(news):
                with (c_news1 if i % 2 == 0 else c_news2):
                    st.markdown(f"""<div class="news-item"><div class="news-source">{item['source']}</div><a class="news-head" href="{item['link']}" target="_blank">{item['title']}</a><div class="news-sum">{item['summary']}</div></div>""", unsafe_allow_html=True)

def render_fundamental_dashboard(user, portfolio, proposals):
    st.title(f"📈 Fundamental Dashboard")
    
    #DOWNLOAD BUTTON
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
    
    st.subheader("Performance vs Market (6 Months)")
    
    # 1. Fetch Data
    bench = fetch_real_benchmark_data(portfolio)
    
    # 2. Plot
    fig = px.line(
        bench, 
        x='Date', 
        y=['SP500', 'TIC_Fund'], 
        color_discrete_map={
            'SP500': '#444444',   # Dark Grey
            'TIC_Fund': '#D4AF37' # TIC Gold
        }
    )
    
    # 3. Critical Styling updates
    fig.update_layout(
        yaxis_title="Value of Investment (Base 100)",
        xaxis_title=None,
        legend_title=None,
        hovermode="x unified" # Shows both values when you hover over one date
    )
    
    # Custom Hover Template to look professional
    fig.update_traces(
        hovertemplate="<b>%{y:.2f}</b>" # Shows 105.20 instead of long decimals
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Allocation")
        fig_pie = px.pie(portfolio, values='target_weight', names='sector', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("Sector Performance")
        # --- FIX: REAL DATA LOGIC ---
        if not portfolio.empty and 'ticker' in portfolio.columns:
            # 1. Get tickers (excluding Cash handled by fetch_live...)
            tickers = portfolio['ticker'].dropna().unique().tolist()
            
            # 2. Fetch Real Change %
            live_data = fetch_live_prices_with_change(tickers)
            
            # 3. Map the 'pct' change to the dataframe
            # We use a lambda to look up the ticker in the live_data dict
            portfolio['real_return'] = portfolio['ticker'].map(
                lambda t: live_data.get(str(t), {}).get('pct', 0.0)
            )
            
            # 4. Plot with Real Data
            fig_tree = px.treemap(
                portfolio, 
                path=[px.Constant("Portfolio"), 'sector', 'ticker'], 
                values='target_weight', 
                color='real_return', # <--- NOW USES REAL DATA
                color_continuous_scale='RdYlGn', # Red to Green
                color_continuous_midpoint=0,     # 0% is the center (Yellow/White)
                hover_data=['real_return']       # Show the % on hover
            )
            
            # Update hover label to look nice
            fig_tree.update_traces(hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<br>Change: %{customdata[0]:.2f}%')
            
            st.plotly_chart(fig_tree, use_container_width=True)
    st.divider()
    st.header("🗳️ Active Proposals")
    current_props = [p for p in proposals if p.get('Dept') == 'Fundamental']
    
    for p in current_props:
        with st.container(border=True):
            c_a, c_b = st.columns([4, 1])
            c_a.subheader(f"{p.get('Type')}: {p.get('Item')}")
            c_a.write(p.get('Description'))
            c_a.caption(f"Closes: {p.get('End_Date')}")
            c_b.metric("Votes", "TBD")

def render_quant_dashboard(user, portfolio, proposals):
    st.title(f"🤖 Quant Lab")
    
    # REPLACED: Moved from simple slider backtest to Monte Carlo Simulation
    with st.expander("🎲 Monte Carlo Risk Engine", expanded=True):
        c_param, c_plot = st.columns([1, 3])
        
        with c_param:
            st.write("**Simulation Settings**")
            n_sims = st.slider("Simulations (N)", 10, 100, 50)
            horizon = st.slider("Time Horizon (Days)", 30, 365, 252)
            # Mean Annual Return
            mu = st.slider("Expected Return (%)", -10, 30, 8) / 100
            # Annual Volatility
            sigma = st.slider("Volatility (%)", 5, 50, 15) / 100
            
            st.info("Uses Geometric Brownian Motion (GBM) to project future portfolio paths.")

        with c_plot:
            # Monte Carlo Logic
            dt = 1/252  # Daily time step
            S0 = 100    # Initial Index Value
            
            # Generate random paths
            # Formula: S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            possible_paths = []
            
            # Create a date range for the x-axis
            dates = pd.date_range(start=datetime.today(), periods=horizon+1)
            
            # We create a dictionary to hold all paths for the dataframe
            sim_data = {'Date': dates}
            
            for i in range(n_sims):
                # Generate random shocks (Z scores)
                Z = np.random.normal(0, 1, horizon)
                
                # Calculate daily returns
                drift = (mu - 0.5 * sigma**2) * dt
                diffusion = sigma * np.sqrt(dt) * Z
                daily_returns = np.exp(drift + diffusion)
                
                # Prepend 1.0 for the start day
                daily_returns = np.insert(daily_returns, 0, 1.0)
                
                # Calculate cumulative price path
                price_path = S0 * np.cumprod(daily_returns)
                sim_data[f'Sim_{i+1}'] = price_path

            # Convert to DataFrame for Plotly
            df_mc = pd.DataFrame(sim_data)
            
            # Melt for easy plotting with Plotly Express
            df_melted = df_mc.melt(id_vars='Date', var_name='Simulation', value_name='Portfolio Value')

            # Plot
            fig = px.line(
                df_melted, 
                x='Date', 
                y='Portfolio Value', 
                color='Simulation',
                title=f"Projected Future Value ({n_sims} Scenarios)",
                color_discrete_sequence=px.colors.qualitative.Pastel # Soft colors
            )
            
            # Style improvements: Hide legend (too messy with 50 lines), add transparency
            fig.update_layout(showlegend=False, xaxis_title="Date", yaxis_title="Indexed Value (Start=100)")
            fig.update_traces(opacity=0.6, line=dict(width=1)) # Make lines thinner and transparent
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Quick Stats
            final_values = df_mc.iloc[-1, 1:] # Last row, excluding Date
            median_val = np.median(final_values)
            worst_case = np.percentile(final_values, 5) # 5th percentile (95% VaR)
            
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.metric("Median Outcome", f"{median_val:.0f}")
            c_s2.metric("Worst Case (5%)", f"{worst_case:.0f}", delta=f"{worst_case-100:.0f}", delta_color="inverse")
            c_s3.metric("Upside (95%)", f"{np.percentile(final_values, 95):.0f}")

    st.divider()
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("Production Models")
        
        st.dataframe(
            # FIX: Slice to keep only the first 4 columns (Model_ID, Allocation, Return, Status)
            portfolio.iloc[:, :4],
            use_container_width=True,
            hide_index=True,
            column_config={
                "model_id": st.column_config.TextColumn("Model Name"),
                "allocation": st.column_config.ProgressColumn(
                    "Allocation",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "ytd_return": st.column_config.NumberColumn(
                    "YTD Return",
                    format="%.1f%%"
                ),
                # Optional: Add nice formatting for the Status column if it exists
                "status": st.column_config.TextColumn("Status")
            }
        )

    with c2:
        st.subheader("Allocation")
        st.plotly_chart(px.pie(portfolio, values='allocation', names='model_id', hole=0.4), use_container_width=True)

    st.divider()

def render_inbox(user, messages, all_members_df):
    st.title("📬 Inbox")
    
    # 1. BOARD/ADMIN: Compose Section
    if user.get('admin', False):
        with st.expander("✍️ Compose Message (Board Only)", expanded=False):
            with st.form("send_msg"):
                # Create smart dropdown options
                # 1. Special Broadcast Groups
                options = ["ALL MEMBERS", "Quant Team", "Fundamental Team"]
                # 2. Add individual users
                individual_users = all_members_df['u'].tolist()
                options += individual_users
                
                target = st.selectbox("To:", options)
                subj = st.text_input("Subject")
                body = st.text_area("Message")
                
                if st.form_submit_button("Send Message"):
                    # Map friendly names to system codes
                    if target == "ALL MEMBERS": final_target = "all"
                    elif target == "Quant Team": final_target = "Quant"
                    elif target == "Fundamental Team": final_target = "Fundamental"
                    else: final_target = target
                    
                    if send_new_message(user['u'], final_target, subj, body):
                        st.success("Message Sent!")
                        st.rerun()
                    else:
                        st.error("Could not write to Excel file.")

    st.divider()

    # 2. EVERYONE: View Messages
    # Filter logic: Show if 'to_user' matches ME, or 'all', or my DEPT
    my_msgs = [
        m for m in messages 
        if m['to_user'] == user['u'] 
        or m['to_user'] == 'all'
        or m['to_user'] == user['d']
    ]
    
    # Sort by date (newest first)
    my_msgs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    if not my_msgs: 
        st.info("No messages in your inbox.")
        return
        
    for m in my_msgs:
        # Visual distinction for Broadcasts vs Direct
        is_broadcast = (m['to_user'] in ['all', 'Quant', 'Fundamental'])
        border_color = "#D4AF37" if not is_broadcast else "rgba(49, 51, 63, 0.2)"
        icon = "📢" if is_broadcast else "📩"
        
        with st.container(border=True):
            c_top, c_time = st.columns([4, 1])
            c_top.markdown(f"**{icon} {m['subject']}**")
            c_time.caption(f"{m['timestamp']}")
            
            st.write(m['body'])
            st.caption(f"From: {m['from_user']} | To: {m['to_user']}")

import streamlit.components.v1 as components

def render_documents(user):
    st.title("📚 Library")
    t1, t2 = st.tabs(["Contract", "Archive"])
    
    with t1: 
        with st.container(border=True):
            st.markdown(f"### Digital Agreement: {user['n']}")
            
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
        # REPLACE THIS with your actual Folder ID
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
            # We use an iframe to show the drive folder directly
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
    members, f_port, q_port, msgs, props, calendar_events, f_total, q_total, df_votes, nav_f, nav_q = load_data()

    if not st.session_state['logged_in']:
        c1, c2, c3 = st.columns([1,1.5,1])
        with c2:
            st.image(TIC_LOGO, width=200)
            st.title("TIC Portal")
            st.info("Welcome to the Internal Management System")
            
            # --- NEW: CLEAN, ALWAYS-VISIBLE LOGIN FORM ---
            with st.form("login_form", clear_on_submit=True):
                st.subheader("Member Login")
                
                # Using keys helps manage state, ensuring inputs don't clear until submit
                u = st.text_input("Username", key="login_u")
                p = st.text_input("Password", type="password", key="login_p")
                
                submitted = st.form_submit_button("Log In", type="primary")
                
                if submitted:
                    user = authenticate(u, p, members)
                    if user is not None:
                        st.session_state['user'] = user.to_dict()
                        st.session_state['logged_in'] = True
                        
                        # Time-aware greeting logic (from previous suggestion)
                        h = datetime.now().hour
                        if 5 <= h < 12: greeting = "Good Morning"
                        elif 12 <= h < 18: greeting = "Good Afternoon"
                        else: greeting = "Good Evening"
                        st.toast(f"🚀 {greeting}, {user['n']}!", icon="👋")
                        
                        st.rerun()
                    else: 
                        st.error("Invalid Username or Password")
            # --- END LOGIN FORM ---
        return
    user = st.session_state['user']
    
    # FETCH PRICES FOR TAPE
    # Get top 15 tickers from fund portfolio to display
    if not f_port.empty and 'ticker' in f_port.columns:
        all_tickers = f_port['ticker'].dropna().unique().tolist()
        live_data = fetch_live_prices_with_change(all_tickers)
        render_ticker_tape(live_data)

    with st.sidebar:
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
        st.caption(f"NAV Fund: €{nav_f:.2f}")
        st.caption(f"NAV Quant: €{nav_q:.2f}")
        # ---------------------------------
        
        menu = ["Simulation", "Inbox", "Library", "Calendar", "Settings"] 

        # Board AND Advisory Board AND Dept Heads see Dashboards
        if user['d'] in ['Fundamental', 'Quant', 'Board', 'Advisory']:
            menu.insert(0, "Risk & Macro")
            menu.insert(0, "Dashboard")
            
        # Fundamental Specific Tools
        if user['d'] == 'Fundamental':
            menu.insert(3, "Valuation Tool")

        # Only show Admin Panel if user has admin=True
        if user.get('admin', False):
            menu.append("Admin Panel")
            
        nav = st.radio("Navigation", menu)
        render_upcoming_events_sidebar(calendar_events)
        
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
    if nav == "Dashboard":
        # 1. BOARD & ADVISORY (See Both)
        if user['d'] in ['Board', 'Advisory']:
            st.title("🏛️ Executive Overview")
            t_fund, t_quant = st.tabs(["📈 Fundamental", "🤖 Quant"])
            
            with t_fund: 
                render_fundamental_dashboard(user, f_port, props)
                st.divider()
                # FIX: Department matches the Tab (Fundamental)
                render_voting_section(user, props, df_votes, "Fundamental")
                
            with t_quant: 
                render_quant_dashboard(user, q_port, props)
                st.divider()
                # FIX: Department matches the Tab (Quant)
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
            
    elif nav == "Risk & Macro": render_risk_macro_dashboard(f_port)
    elif nav == "Valuation Tool": render_valuation_sandbox() 
    elif nav == "Simulation": 
        t_sim, t_lead = st.tabs(["🎮 Trade", "🏆 Leaderboard"])
        with t_sim: render_simulation(user)
        with t_lead: render_leaderboard(user) 
    elif nav == "Calendar": render_calendar_view(user, calendar_events)
    elif nav == "Inbox": render_inbox(user, msgs, members)
    elif nav == "Library": render_documents(user)
    elif nav == "Settings": render_offboarding(user)
    elif nav == "Admin Panel": render_admin_panel(user, members, f_port, q_port, f_total + q_total, props, df_votes, nav_f, nav_q)

    # PROFESSIONAL FOOTER
    st.markdown("---")
    c_foot1, c_foot2 = st.columns(2)
    with c_foot1:
        st.caption("© 2025 Tilburg Investment Club | Internal Portal v1.0")
    with c_foot2:
        st.caption("Data provided by Yahoo Finance & TIC Research Team")

if __name__ == "__main__":
    main()























