import pandas as pd
from ib_insync import *
import gspread
from google.oauth2.service_account import Credentials
import toml
import time
import yfinance as yf

# --- CONFIGURATION ---
IBKR_PORT = 7496 
IBKR_IP = '127.0.0.1'
CLIENT_ID = 999
FUNDAMENTAL_ACCT = 'U11415280'
QUANT_ACCT = 'U13197848'
SECRETS_FILE = ".streamlit/secrets.toml"

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# --- 1. INTELLIGENT MAPPING LOGIC ---
def get_yahoo_ticker(ib_ticker, primary_exchange, currency):
    """
    Converts IBKR Ticker + Exchange into Yahoo Finance Ticker.
    """
    # 1. Handle Empty/None Exchange
    if not primary_exchange:
        primary_exchange = ""
    
    px = primary_exchange.upper()
    
    # 2. Logic Map (IBKR Exchange Code -> Yahoo Suffix)
    if px in ['IBIS', 'IBIS2', 'GER', 'FWB']: 
        return f"{ib_ticker}.DE"   # Germany (Xetra/Frankfurt)
    
    elif px in ['AEB', 'EBS', 'EOE']: 
        return f"{ib_ticker}.AS"   # Amsterdam
        
    elif px in ['SBF', 'FRANCE', 'MATIF', 'MONEP']: 
        return f"{ib_ticker}.PA"   # Paris
        
    elif px in ['SB', 'BELFOX', 'BRUSSELS']: 
        return f"{ib_ticker}.BR"   # Brussels
        
    elif px in ['LSE', 'LSEETF', 'LSE_ETF']: 
        return f"{ib_ticker}.L"    # London
        
    elif px in ['EBS', 'SWISS', 'VIRT-X']: 
        return f"{ib_ticker}.SW"   # Switzerland
        
    elif px in ['BVCH', 'MI', 'MTA']: 
        return f"{ib_ticker}.MI"   # Milan
    
    elif px in ['BM', 'MADRID']: 
        return f"{ib_ticker}.MC"   # Madrid

    # 3. US Stocks (NASDAQ, NYSE, AMEX) -> No Suffix
    # If currency is USD and exchange is SMART/ISLAND, assume US.
    if currency == 'USD' and px in ['SMART', 'ISLAND', 'NYSE', 'NASDAQ', 'ARCA', 'AMEX', 'BATS']:
        return ib_ticker

    # 4. Default: Return original if no rule matches
    return ib_ticker

# --- 1. QUICK SECTOR MAP (Add more as you find them) ---
# This saves time so you don't have to fetch from Yahoo every single run
SECTOR_CACHE = {
    "NVDA": "Technology",
    "MSFT": "Technology",
    "PLTR": "Software",
    "DIE.BR": "Consumer Cyclical",
    "ENR.DE": "Industrials",
    "INGA.AS": "Financial Services",
    "CASH_EUR": "Liquidity",
    "CASH_USD": "Liquidity"
}

def get_sector(ticker):
    """Retrieves sector from cache or live from Yahoo Finance."""
    # 1. Check our quick list first
    if ticker in SECTOR_CACHE:
        return SECTOR_CACHE[ticker]
    
    # 2. If not found, ask Yahoo Finance
    try:
        print(f"🔍 Looking up sector for {ticker}...")
        info = yf.Ticker(ticker).info
        sector = info.get('sector', 'Other')
        # Store it so we don't ask again this session
        SECTOR_CACHE[ticker] = sector
        return sector
    except:
        return "Other"

def get_gspread_client():
    try:
        with open(SECRETS_FILE, "r") as f:
            secrets = toml.load(f)
        creds_info = secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        print(f"❌ Google Auth Error: {e}")
        return None

def fetch_portfolio_data():
    ib = IB()
    try:
        print(f"🔌 Connecting to TWS on port {IBKR_PORT}...")
        ib.connect(IBKR_IP, IBKR_PORT, clientId=CLIENT_ID)
        print("✅ Connected!")

        ib.client.reqAccountUpdates(True, FUNDAMENTAL_ACCT)
        ib.client.reqAccountUpdates(True, QUANT_ACCT)
        ib.sleep(2)

        # 1. Fetch RAW Positions
        all_positions = ib.positions()
        all_values = ib.accountValues()
        
        print(f"   Stocks found: {len(all_positions)}")
        
        # 2. QUALIFY CONTRACTS (The Fix for '0' Exchange)
        # We need the full contract details to get the listing exchange
        contracts = [p.contract for p in all_positions if p.contract.secType == 'STK']
        print(f"   Qualifying {len(contracts)} contracts to find exchanges...")
        ib.qualifyContracts(*contracts) # Forces IB to find the PrimaryExchange
        
        fund_rows = []
        quant_rows = []
        
        # --- PROCESS STOCKS ---
        for p in all_positions:
            if p.contract.secType != 'STK': continue 

            # Use primaryExchange if available, otherwise fallback to exchange
            exchange = p.contract.primaryExchange or p.contract.exchange
            raw_ticker = p.contract.symbol
            currency = p.contract.currency
            
            # Apply the intelligent suffix logic from before
            clean_ticker = get_yahoo_ticker(raw_ticker, exchange, currency)
            
            row = {
                "Ticker": clean_ticker,
                "Name": p.contract.localSymbol,
                "Sector": get_sector(clean_ticker),
                "Units": p.position,
                "Type": "Equity",
                "Target_Weight": 0.0,
                "Exchange": exchange,   # Should now show 'AEB', 'IBIS', etc.
                "Model": "Live_Broker" if p.account == QUANT_ACCT else ""
            }
            
            if p.account == FUNDAMENTAL_ACCT:
                del row["Model"]
                fund_rows.append(row)
            elif p.account == QUANT_ACCT:
                quant_rows.append(row)

        # --- PROCESS CASH ---
        for item in all_values:
            if item.tag == 'TotalCashBalance':
                if item.currency == 'BASE': continue
                try: amount = float(item.value)
                except: continue
                if amount < 1.0: continue 
                
                row = {
                    "Ticker": f"CASH_{item.currency}",
                    "Name": f"{item.currency} Cash",
                    "Sector": "Liquidity",
                    "Units": amount,
                    "Type": "Cash",
                    "Target_Weight": 0.0,
                    "Exchange": "FOREX", 
                    "Model": "Cash" if item.account == QUANT_ACCT else ""
                }
                
                if item.account == FUNDAMENTAL_ACCT:
                    del row["Model"]
                    fund_rows.append(row)
                elif item.account == QUANT_ACCT:
                    quant_rows.append(row)

        ib.disconnect()
        
        # Define Columns (Added 'Exchange')
        cols_fund = ['Ticker', 'Name', 'Sector', 'Units', 'Type', 'Target_Weight', 'Exchange']
        cols_quant = ['Ticker', 'Name', 'Sector', 'Units', 'Type', 'Target_Weight', 'Exchange', 'Model']
        
        df_fund = pd.DataFrame(fund_rows)
        if not df_fund.empty: df_fund = df_fund[cols_fund]
            
        df_quant = pd.DataFrame(quant_rows)
        if not df_quant.empty: df_quant = df_quant[cols_quant]
            
        return {'Fundamentals': df_fund, 'Quant': df_quant}

    except Exception as e:
        print(f"❌ IBKR Connection Failed: {e}")
        return None

def push_to_gsheet(sheet_name, df):
    if df.empty:
        print(f"⚠️ No data for {sheet_name}. Skipping update.")
        return

    client = get_gspread_client()
    if not client: return

    try:
        sh = client.open("TIC_Database_Master")
        ws = sh.worksheet(sheet_name)
        
        print(f"💾 Overwriting '{sheet_name}' with {len(df)} rows...")
        ws.clear()
        headers = df.columns.tolist()
        data_to_write = [headers] + df.values.tolist()
        ws.update(range_name="A1", values=data_to_write)
        print(f"✅ {sheet_name} Updated.")
        
    except Exception as e:
        print(f"🔥 Write Error ({sheet_name}): {e}")

if __name__ == "__main__":
    print("🚀 Starting Smart Suffix Sync...")
    sorted_data = fetch_portfolio_data()
    
    if sorted_data:
        push_to_gsheet("Fundamentals", sorted_data['Fundamentals'])
        push_to_gsheet("Quant", sorted_data['Quant'])
    
    print("\n🏁 Sync Complete.")