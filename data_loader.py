import time
import json
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, date
import os
import toml 
import math
import concurrent.futures

# --- CONFIGURATION ---
UPDATE_INTERVAL = 300  # 5 minutes (Prices & Sheets)
EVENTS_INTERVAL = 43200 # 12 Hours (Earnings Dates)
MARKET_FILE = "market_snapshot.json"
DB_FILE = "database_snapshot.json"

# --- GOOGLE AUTH SETUP ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_gspread_client():
    """Reads secrets.toml and authenticates with Google."""
    try:
        with open(".streamlit/secrets.toml", "r") as f:
            secrets = toml.load(f)
        creds_info = secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return None

def fetch_database_snapshot():
    """Fetches ALL tabs from Google Sheets."""
    print("📥 Fetching Google Sheets Database...")
    client = get_gspread_client()
    if not client: return None
    
    snapshot = {}
    try:
        sheet = client.open("TIC_Database_Master")
        # List of all tabs we need
        tabs = ["Fundamentals", "Quant", "Members", "Events", "Proposals", "Votes", "Attendance", "Expenses"]
        
        for tab in tabs:
            try:
                ws = sheet.worksheet(tab)
                data = ws.get_all_values()
                snapshot[tab] = data
                print(f"   - Fetched {tab} ({len(data)} rows)")
            except Exception as e:
                print(f"   ⚠️ Could not fetch {tab}: {e}")
                snapshot[tab] = []
                
        snapshot["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return snapshot
        
    except Exception as e:
        print(f"🔥 Google Sheet Error: {e}")
        return None

TICKER_MAP = {
    "ADYEN": "ADYEN.AS",  # Amsterdam
    "PRX": "PRX.AS",      # Prosus (Amsterdam)
    "INGA": "INGA.AS",    # ING (Amsterdam)
    "FLOW": "FLOW.AS",    # Flow Traders (Amsterdam)
    "VLK": "VLK.AS",      # Van Lanschot (Amsterdam) - Verify if this is the correct VLK
    "RWE": "RWE.DE",      # RWE AG (Germany)
    "ENR": "ENR.DE",      # Siemens Energy (Germany)
    "HEI": "HEI.DE",      # Heidelberg Materials
    "DIE": "DIE.BR",      # D'Ieteren (Brussels)
    "AGS": "AGS.BR",      # Ageas (Brussels)
    "UMI": "UMI.BR",      # Umicore (Brussels)
    "ENGI": "ENGI.PA",    # Engie (Paris)
    "AIR": "AIR.PA",      # Airbus (Paris)
    "RR.": "RR.L",        # Rolls Royce (London) - Yahoo often uses RR.L
    "BFT": "BFT.BR",      # Belfius? Or check specific holding.
    "IOC": "IOC.L",       # Check if this is Cosmos or similar.
    "NURS": "NURS.DE",    # Check specific listing
}

def extract_tickers_from_snapshot(db_snapshot):
    """Helper to get a clean list of unique tickers from the loaded DB."""
    tickers = set()
    
    def get_from_tab(tab_name, col_options):
        data = db_snapshot.get(tab_name, [])
        if len(data) < 2: return
        headers = [h.lower() for h in data[0]]
        
        idx = -1
        for opt in col_options:
            if opt in headers:
                idx = headers.index(opt)
                break
        
        if idx != -1:
            for row in data[1:]:
                if len(row) > idx:
                    raw_t = row[idx].strip().upper()
                    
                    # CLEANUP LOGIC
                    if raw_t and "CASH" not in raw_t and "EUR" not in raw_t:
                        # 1. Check if we have a manual map for it
                        if raw_t in TICKER_MAP:
                            mapped_t = TICKER_MAP[raw_t]
                            tickers.add(mapped_t)
                        # 2. Heuristic: If it looks like a US ticker (no dots), keep it.
                        #    If it needs a suffix but isn't in map, it might fail (add to map later).
                        else:
                            tickers.add(raw_t)

    get_from_tab("Fundamentals", ["ticker"])
    get_from_tab("Quant", ["ticker", "model_id"])
    return list(tickers)

def fetch_single_calendar_event(t):
    """Helper: Fetches one ticker's earnings date (runs in parallel)."""
    try:
        if not isinstance(t, str): return None
        stock = yf.Ticker(t)
        cal = stock.calendar
        if cal and 'Earnings Date' in cal:
            dates = cal['Earnings Date']
            if dates:
                next_date = dates[0]
                if next_date >= date.today():
                    return {
                        'title': f"{t} Earnings",
                        'ticker': t,
                        'date': next_date.strftime('%Y-%m-%d'),
                        'type': 'market',
                        'audience': 'all'
                    }
    except:
        return None
    return None

def fetch_market_events_parallel(ticker_list):
    """Runs the heavy earnings fetch in parallel threads."""
    print(f"📅 Updating Earnings Calendar for {len(ticker_list)} tickers...")
    events = []
    
    # Safe limit
    safe_tickers = ticker_list[:50] 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single_calendar_event, t): t for t in safe_tickers}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                events.append(res)
    
    print(f"✅ Found {len(events)} upcoming events.")
    return events

def fetch_market_prices(ticker_list):
    """Fetches Prices for tickers found in the DB snapshot."""
    # Add Indices/Forex manually
    full_list = set(ticker_list)
    full_list.update(["EURUSD=X", "^GSPC", "^VIX", "BTC-USD", "JPYEUR=X", "GBPEUR=X"])
    final_list = list(full_list)
    
    print(f"💹 Fetching Prices for {len(final_list)} assets...")
    
    market_snap = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prices": {}
    }
    
    if final_list:
        try:
            # Threaded=False is safer for stability, though slightly slower
            data = yf.download(final_list, period="5d", interval="1d", progress=False, group_by='ticker', threads=False)
            
            for t in final_list:
                try:
                    df = data[t] if len(final_list) > 1 else data
                    df = df.dropna(subset=['Close'])

                    if not df.empty:
                        current = float(df['Close'].iloc[-1])
                        prev = float(df['Close'].iloc[-2]) if len(df) > 1 else current
                        change = current - prev
                        pct = (change / prev) * 100 if prev != 0 else 0.0
                        
                        # Sanity cleanup
                        if math.isnan(current): current = 0.0
                        if math.isnan(change): change = 0.0
                        if math.isnan(pct): pct = 0.0

                        market_snap["prices"][t] = {"price": current, "change": change, "pct": pct}
                    else:
                        market_snap["prices"][t] = {"price": 0.0, "change": 0.0, "pct": 0.0}
                except:
                    market_snap["prices"][t] = {"price": 0.0, "change": 0.0, "pct": 0.0}
        except Exception as e:
            print(f"   ⚠️ Price Fetch Error: {e}")

    return market_snap

def save_json(data, filename):
    temp = filename + ".tmp"
    with open(temp, 'w') as f:
        json.dump(data, f)
    os.replace(temp, filename)
    print(f"💾 Saved {filename}")

def main():
    print("🚀 FULL STACK Data Loader Started...")
    
    # State variables for the "Once a Day" logic
    cached_events = [] 
    last_events_update = 0
    
    while True:
        loop_start = time.time()
        
        # 1. ALWAYS Fetch DB (Google Sheets) to get new tickers/members
        db_data = fetch_database_snapshot()
        
        if db_data:
            # 2. Extract Tickers
            current_tickers = extract_tickers_from_snapshot(db_data)
            
            # 3. CONDITIONAL: Fetch Earnings Events (Only every 12 hours)
            time_since_last = time.time() - last_events_update
            if time_since_last > EVENTS_INTERVAL or not cached_events:
                cached_events = fetch_market_events_parallel(current_tickers)
                last_events_update = time.time()
            else:
                print(f"⏳ Skipping Earnings Fetch (Next update in {int((EVENTS_INTERVAL - time_since_last)/60)} min)")

            # 4. INJECT Events into the DB Snapshot
            db_data["Market_Events"] = cached_events
            
            # 5. Save DB Snapshot (Contains Sheets Data + Market Events)
            save_json(db_data, DB_FILE)
            
            # 6. ALWAYS Fetch Market Prices (Real-time)
            market_data = fetch_market_prices(current_tickers)
            if market_data:
                save_json(market_data, MARKET_FILE)
            
        # Smart Sleep (Account for processing time)
        elapsed = time.time() - loop_start
        sleep_time = max(10, UPDATE_INTERVAL - elapsed)
        
        print(f"💤 Sleeping for {int(sleep_time)}s...\n" + "-"*40)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()