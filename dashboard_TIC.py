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
    clean_tickers = list(set([str(t) for t in tickers if isinstance(t, str)]))
    
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
def fetch_real_benchmark_data(portfolio_df):
    """
    Fetches real historical data for the S&P 500 and the user's portfolio.
    Constructs a normalized index (Start = 100) for comparison.
    """
    end = datetime.now()
    start = end - timedelta(days=180) # 6 Month lookback
    
    # 1. Fetch S&P 500 (The Benchmark)
    try:
        # ^GSPC is the ticker for S&P 500
        sp500 = yf.download('^GSPC', start=start, end=end, progress=False)['Close']
        # Normalize: (Price / Start_Price) * 100
        sp500_norm = (sp500 / sp500.iloc[0]) * 100
        
        # Prepare the main dataframe
        df_chart = pd.DataFrame({'Date': sp500.index, 'SP500': sp500_norm.values})
    except Exception as e:
        # Fallback if offline
        dates = pd.date_range(start=start, end=end)
        return pd.DataFrame({'Date': dates, 'SP500': 100, 'TIC_Fund': 100})

    # 2. Fetch Portfolio Performance (The TIC Fund)
    if not portfolio_df.empty and 'ticker' in portfolio_df.columns:
        try:
            tickers = portfolio_df['ticker'].dropna().unique().tolist()
            # Clean tickers (remove non-strings)
            tickers = [t for t in tickers if isinstance(t, str)]
            
            if tickers:
                # Download all stocks at once
                data = yf.download(tickers, start=start, end=end, progress=False)['Close']
                
                # Forward fill missing data (e.g. holidays) to prevent calculation breaks
                data = data.ffill()
                
                # Calculate "Pro-Forma" History:
                # We assume the CURRENT weights were held for the whole period.
                # (This is a standard way to visualize "Strategy Performance")
                
                # Get weights from Excel, map them to the tickers
                # If 'target_weight' is missing, assume equal weight
                if 'target_weight' in portfolio_df.columns:
                    weights = portfolio_df.set_index('ticker')['target_weight']
                    # Normalize weights to sum to 1.0 just in case
                    weights = weights / weights.sum()
                else:
                    weights = pd.Series(1/len(tickers), index=tickers)
                
                # Calculate Weighted Index
                # 1. Normalize all stocks to start at 100
                norm_data = (data / data.iloc[0]) * 100
                
                # 2. Apply weights (Multiply each column by its weight)
                # We align the weights to the columns available in the downloaded data
                valid_cols = norm_data.columns.intersection(weights.index)
                weighted_index = norm_data[valid_cols].multiply(weights[valid_cols]).sum(axis=1)
                
                # 3. Add to the main dataframe (Aligning dates)
                # We reindex to match the SP500 dates to handle slight mismatches
                aligned_tic = weighted_index.reindex(df_chart['Date'], method='ffill')
                df_chart['TIC_Fund'] = aligned_tic.values
            else:
                df_chart['TIC_Fund'] = df_chart['SP500'] # Fallback line
        except Exception as e:
            print(f"Portfolio History Error: {e}")
            df_chart['TIC_Fund'] = 100 # Flat line on error
    else:
        df_chart['TIC_Fund'] = 100

    return df_chart

@st.cache_data
def load_data():
    # --- 1. CONFIGURATION & MAPPING ---
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

    members_list = []

    # --- 2. LOAD EXTERNAL MEMBER LIST ---
    file_path = "data/Member List.xlsx"

    if os.path.exists(file_path):
        try:
            df_external = pd.read_excel(file_path, engine='openpyxl')

            for index, row in df_external.iterrows():
                role_code = str(row.get('Role', 'other')).strip().lower()
                role_data = ROLE_MAP.get(role_code, {'r': 'Member', 'd': 'General', 's': 'General', 'admin': False})

                full_name = str(row.get('Name', 'Unknown')).strip()
                username = full_name.lower().replace(" ", ".")
                email = str(row.get('Email', f"{username}@tilburg.edu")).strip()
                if email == 'nan' or email == '': email = f"{username}@tilburg.edu"
                
                # FIX: Safely handle NaN values (float) before conversion to int(0 or 1)
                raw_liq_val = row.get('Liq Pending', 0)
                
                if pd.isna(raw_liq_val):
                    liq_pending_val = 0
                else:
                    liq_pending_val = int(raw_liq_val)

                member = {
                    'u': username,
                    'p': str(row.get('Password', 'pass')).strip(),
                    'n': full_name,
                    'email': email,
                    'r': role_data['r'],
                    'd': role_data['d'],
                    's': role_data['s'],
                    'admin': role_data.get('admin', False),
                    'status': 'Pending' if liq_pending_val == 1 else 'Active',
                    'liq_pending': liq_pending_val, 
                    'contribution': float(row.get('Initial Investment', 0)),
                    'value': float(row.get('Current value', 0)),
                    'contract_text': "TIC MEMBERSHIP AGREEMENT 2024-2025..."
                }
                members_list.append(member)
        except Exception as e:
            st.error(f"Error reading member list: {e}")
            members_list = [{'u': 'admin', 'p': 'pass', 'n': 'System Admin', 'r': 'Admin', 'd': 'Board', 's': 'Tech', 'admin': True, 'contribution': 0, 'value': 0}]
    else:
        st.warning(f"Member file not found at: {file_path}")
        members_list = [{'u': 'flavian', 'p': 'pass', 'n': 'Flavian', 'r': 'President', 'd': 'Board', 's': 'Management', 'admin': True, 'contribution': 2500, 'value': 2950}]

    members = pd.DataFrame(members_list)

    # --- 3. OTHER DATA (Portfolios) ---
    portfolio_path = "data/TIC_Portfolios.xlsx"
    
    fund_portfolio = pd.DataFrame(columns=['ticker', 'name', 'sector', 'target_weight', 'total'])
    quant_portfolio = pd.DataFrame(columns=['model_id', 'allocation', 'ytd_return', 'total'])
    
    # NEW: Variables to store totals
    f_total = 0.0
    q_total = 0.0

    if os.path.exists(portfolio_path):
        try:
            # Read Fundamentals
            fund_portfolio = pd.read_excel(portfolio_path, sheet_name='Fundamentals', engine='openpyxl')
            fund_portfolio.columns = fund_portfolio.columns.str.lower()
            # Extract Total from first row
            if 'total' in fund_portfolio.columns:
                val = fund_portfolio['total'].iloc[0]
                if not pd.isna(val): f_total = float(val)

            # Read Quant
            quant_portfolio = pd.read_excel(portfolio_path, sheet_name='Quant', engine='openpyxl')
            quant_portfolio.columns = quant_portfolio.columns.str.lower()
            # Extract Total from first row
            if 'total' in quant_portfolio.columns:
                val = quant_portfolio['total'].iloc[0]
                if not pd.isna(val): q_total = float(val)
                
        except Exception as e:
            st.error(f"Error reading Portfolio Excel: {e}")
    else:
        st.warning("Portfolio file not found. Using empty data.")

    # --- 4. MESSAGES & EVENTS ---
    messages = []
    msg_path = "data/TIC_Messages.xlsx"
    if os.path.exists(msg_path):
        try:
            df_msg = pd.read_excel(msg_path, engine='openpyxl')
            # Convert DataFrame to list of dicts for the app
            messages = df_msg.to_dict('records')
            
            # Standardize keys to lowercase just in case
            messages = [{k.lower(): v for k, v in m.items()} for m in messages]
        except:
            messages = []
    
    proposals = [
        {'id': 101, 'dept': 'Fundamental', 'item': 'AMD', 'type': 'BUY', 'desc': 'Undervalued competitor.', 'end': '2023-12-10', 'for': 3, 'against': 0},
        {'id': 102, 'dept': 'Quant', 'item': 'Kill RMS-Prop', 'type': 'MODEL_CHANGE', 'desc': 'Underperformance.', 'end': '2023-12-15', 'for': 1, 'against': 1}
    ]
    
   # --- DYNAMIC CALENDAR GENERATION ---
    today = datetime.now()
    
    # 1. Fetch Real Market Events (Earnings)
    real_market_events = []
    if not fund_portfolio.empty and 'ticker' in fund_portfolio.columns:
        fund_tickers = fund_portfolio['ticker'].dropna().unique().tolist()
        real_market_events = fetch_company_events(fund_tickers)

    # 2. Load Internal/Macro Events from Excel
    events_path = "data/TIC_Events.xlsx"
    manual_events = []
    
    if os.path.exists(events_path):
        try:
            df_events = pd.read_excel(events_path, engine='openpyxl')
            # Standardize date format to string 'YYYY-MM-DD'
            df_events['Date'] = pd.to_datetime(df_events['Date']).dt.strftime('%Y-%m-%d')
            
            for _, row in df_events.iterrows():
                manual_events.append({
                    'title': str(row['Title']),
                    'ticker': str(row['Ticker']),
                    'date': str(row['Date']),
                    'type': str(row['Type']).lower(),
                    'audience': str(row['Audience'])
                })
        except Exception as e:
            st.error(f"Error reading Events Excel: {e}")
    
    # 3. Combine All
    full_calendar = real_market_events + manual_events

    return members, fund_portfolio, quant_portfolio, messages, proposals, full_calendar, f_total, q_total
# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
# Define the absolute path for the member list, consistent with the user's setup
MEMBER_FILE_PATH = "data/Member List.xlsx"
def update_member_fields_in_excel_bulk(usernames, updates_dict):
    """Reads the member file, updates multiple fields for multiple users, and saves the file."""
    if not os.path.exists(MEMBER_FILE_PATH):
        st.error("Error: Cannot find member file to update.")
        return False
        
    try:
        df = pd.read_excel(MEMBER_FILE_PATH, engine='openpyxl')
        
        # 1. Create a transient 'u' (username) column for matching
        df['u'] = df['Name'].astype(str).str.lower().str.strip().str.replace(' ', '.')
        
        # 2. Iterate through all target usernames
        for username in usernames:
            user_index = df[df['u'] == username].index
            
            if not user_index.empty:
                # 3. Apply the dictionary of updates (e.g., {'Liq Pending': 0, 'Liq approved': 1})
                for field, value in updates_dict.items():
                    df.loc[user_index, field] = value
        
        # 4. Clean up and save
        df = df.drop(columns=['u'])
        df.to_excel(MEMBER_FILE_PATH, index=False, engine='openpyxl')
        st.cache_data.clear() 
        return True
    except Exception as e:
        # Catch file locked error
        st.error(f"Error updating member Excel in bulk: {e}. Is the file open?")
        return False
    
def update_member_field_in_excel(username, field, value, file_path=MEMBER_FILE_PATH):
    """
    Reads the member file, updates a single user's field, and saves the file.
    Note: Requires the user to re-log or manually trigger cache clear to see the change reflected fully.
    """
    if not os.path.exists(file_path):
        st.error("Error: Cannot find member file to update.")
        return False
        
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # 1. Create a transient 'u' (username) column for matching the logged-in user
        # This matches the logic used in load_data
        df['u'] = df['Name'].astype(str).str.lower().str.strip().str.replace(' ', '.')
        
        # 2. Find the row index of the user
        user_index = df[df['u'] == username].index
        
        if not user_index.empty:
            # 3. Update the required field (e.g., 'Liq Pending')
            df.loc[user_index, field] = value
            
            # 4. Remove transient column and save
            df = df.drop(columns=['u'])
            df.to_excel(file_path, index=False, engine='openpyxl')
            
            # IMPORTANT: We must clear the data cache so load_data re-reads the file next time.
            st.cache_data.clear() 
            return True
        return False

    except Exception as e:
        st.error(f"Error updating member Excel: {e}. Is the file open in Excel?")
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

def render_admin_panel(user, members_df, f_port, q_port, total_aum):
    st.title("🔒 Admin Console")
    st.info(f"Logged in as: {user['n']} ({user['r']})")
    
    # Define Path for saving member list
    MEMBER_FILE_PATH = "data/Member List.xlsx"
    
    # 1. TABS
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Member Database", "💰 Treasury", "📄 Reporting", "✅ Attendance"])
    
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
                if update_member_fields_in_excel_bulk(selected_usernames, updates):
                    st.success(f"✅ Approved {len(selected_usernames)} requests. Please clear cache (or restart) to update.")
                else:
                    st.error("❌ Approval failed. Check file permissions.")
                st.rerun()

            # DENY BUTTON LOGIC
            if c_b.button("Deny Selected", disabled=not selected_usernames):
                # DENY: Liq Pending = 0, Liq Approved = 0 (Clears pending state)
                updates = {'Liq Pending': 0, 'Liq Approved': 0}
                if update_member_fields_in_excel_bulk(selected_usernames, updates):
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
                if update_member_field_in_excel(user['u'], 'Liq Pending', 1):
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
                if update_member_field_in_excel(user['u'], 'Liq Pending', 0):
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
    with st.expander("📥 Export Data"):
        csv = portfolio.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Portfolio (CSV)",
            csv,
            "tic_fundamentals.csv",
            "text/csv",
            key='download-fund'
        )
    
    st.subheader("Performance vs Market (6 Months)")
    bench = fetch_real_benchmark_data(portfolio)
    st.plotly_chart(
        px.line(
            bench, 
            x='Date', 
            y=['SP500', 'TIC_Fund'], 
            color_discrete_map={
                'SP500': '#444444',  # Dark Grey for Benchmark
                'TIC_Fund': '#D4AF37' # Gold for Your Fund
            },
            labels={'value': 'Rebased Performance (100)', 'variable': 'Index'}
        ), 
        use_container_width=True
    )
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Allocation")
        fig_pie = px.pie(portfolio, values='target_weight', names='sector', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("Sector Performance")
        portfolio['mock_return'] = np.random.uniform(-15, 25, size=len(portfolio))
        fig_tree = px.treemap(
            portfolio, path=[px.Constant("Portfolio"), 'sector', 'ticker'], 
            values='target_weight', color='mock_return',
            color_continuous_scale='RdYlGn', color_continuous_midpoint=0
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()
    st.header("🗳️ Active Proposals")
    props = [p for p in proposals if p['dept'] == 'Fundamental']
    for p in props:
        with st.container(border=True):
            c_a, c_b = st.columns([4, 1])
            c_a.subheader(f"{p['type']} {p['item']}")
            c_a.write(p['desc'])
            c_a.caption(f"Closes: {p['end']}")
            c_b.metric("Votes", p['for'])
            if c_b.button("Vote YES", key=f"f_{p['id']}"): st.toast("Vote Recorded")

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
    st.header("🗳️ Governance")
    props = [p for p in proposals if p['dept'] == 'Quant']
    for p in props:
        with st.container(border=True):
            st.subheader(f"⚙️ {p['item']}")
            st.write(p['desc'])
            if st.button("Deploy", key=f"q_{p['id']}"): st.toast("Deployment Triggered")

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
    members, f_port, q_port, msgs, props, calendar_events, f_total, q_total = load_data()

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
        st.header(user['n'])
        st.caption(f"{user['r']} | {user['email']}")
        
        # FINANCIAL SUMMARY IN SIDEBAR
        val = user.get('value', 0)
        cont = user.get('contribution', 1) # avoid div/0 error
        growth_pct = ((val - cont) / cont) * 100 if cont > 0 else 0.0
        
        st.metric("My Stake", f"€{val:,.0f}", f"{growth_pct:.1f}%")
        
        menu = ["Simulation", "Inbox", "Library", "Settings", "Calendar"] 

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
        if user['d'] in ['Board', 'Advisory']:
            st.title("🏛️ Executive Overview")
            t_fund, t_quant = st.tabs(["📈 Fundamental", "🤖 Quant"])
            with t_fund: render_fundamental_dashboard(user, f_port, props)
            with t_quant: render_quant_dashboard(user, q_port, props)
        elif user['d'] == 'Quant': 
            render_quant_dashboard(user, q_port, props)
        else: 
            render_fundamental_dashboard(user, f_port, props)
            
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
    elif nav == "Admin Panel": render_admin_panel(user, members, f_port, q_port, total_aum  =f_total + q_total)

if __name__ == "__main__":
    main()
