import requests
import json
import time
import os
import datetime

# Full list of S&P 500 stock tickers (replace with actual full list)
SP500_TICKERS = [
    "AAPL", "NVDA", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "AVGO", "BRK.B",
    "WMT", "JPM", "LLY", "V", "MA", "ORCL", "UNH", "COST", "XOM", "NFLX",
    "HD", "PG", "JNJ", "BAC", "ABBV", "CRM", "TMUS", "KO", "CVX", "PLTR",
    "WFC", "CSCO", "ACN", "IBM", "PM", "ABT", "GE", "MS", "MCD", "AXP",
    "LIN", "GS", "ISRG", "MRK", "TMO", "NOW", "BX", "ADBE", "DIS", "PEP",
    "QCOM", "T", "AMD", "VZ", "CAT", "UBER", "BKNG", "TXN", "SPGI", "INTU",
    "RTX", "C", "AMGN", "BSX", "PGR", "BLK", "UNP", "DHR", "SCHW", "SYK",
    "PFE", "LOW", "NEE", "TJX", "BA", "AMAT", "ANET", "CMCSA", "HON", "PANW",
    "FI", "DE", "GILD", "SBUX", "ADP", "KKR", "COP", "ETN", "MDT", "VRTX",
    "PLD", "MMC", "CRWD", "MU", "BMY", "NKE", "ADI", "LRCX", "CB", "INTC",
    "GEV", "ABNB", "KLAC", "LMT", "CEG", "UPS", "WELL", "ICE", "MCO", "SO",
    "APO", "WM", "EQIX", "MO", "ELV", "PH", "SHW", "AMT", "CME", "DUK",
    "FTNT", "AON", "APH", "CVS", "CTAS", "TT", "CI", "CDNS", "SNPS", "MMM",
    "AJG", "HCA", "DELL", "MDLZ", "MAR", "CMG", "PYPL", "COF", "PNC", "ITW",
    "ORLY", "ECL", "USB", "MCK", "TDG", "MSI", "EOG", "RSG", "REGN", "RCL",
    "ZTS", "APD", "CL", "SPG", "EMR", "WMB", "WDAY", "GD", "ADSK", "BDX",
    "FDX", "CSX", "HLT", "NOC", "BK", "TFC", "ROP", "JCI", "KMI", "TGT",
    "SLB", "AZO", "NSC", "CPRT", "OKE", "CHTR", "VST", "NXPI", "AFL", "FCX",
    "CARR", "DLR", "MET", "PCAR", "HWM", "AEP", "TRV", "SRE", "PAYX", "PSX",
    "AMP", "NEM", "PSA", "AXON", "CMI", "GWW", "ALL", "MPC", "DFS", "URI",
    "GM", "MNST", "O", "KR", "COR", "D", "NDAQ", "AIG", "BKR", "FANG",
    "ROST", "TEL", "OXY", "HES", "GLW", "EW", "CTSH", "TRGP", "LULU", "MSCI",
    "ODFL", "KMB", "FICO", "CTVA", "CBRE", "EXC", "KDP", "FAST", "VLO", "AME",
    "GEHC", "PWR", "KVUE", "DAL", "PEG", "VRSK", "YUM", "DHI", "GRMN", "IT",
    "PRU", "XEL", "A", "OTIS", "CCI", "F", "LHX", "TTWO", "FIS", "IDXX",
    "VMC", "LYV", "ETR", "DXCM", "DD", "SYY", "KHC", "IR", "CCL", "UAL",
    "RMD", "EXR", "IQV", "EA", "PCG", "EBAY", "ROK", "ACGL", "MTB", "ED",
    "LEN", "MPWR", "RJF", "WEC", "GIS", "NUE", "MLM", "HIG", "WAB", "VICI",
    "WTW", "HSY", "EQT", "KEYS", "BRO", "HPQ", "TPL", "XYL", "LVS", "AVB",
    "EFX", "HUM", "TSCO", "CAH", "CSGP", "MCHP", "ANSS", "IP", "FITB", "STZ",
    "VTR", "STT", "HPE", "K", "CNC", "BR", "SMCI", "IRM", "EQR", "SW",
    "DOV", "FTV", "TYL", "PPG", "DOW", "DTE", "MTD", "AEE", "GDDY", "CPAY",
    "GPN", "EXPE", "CHD", "WBD", "SYF", "CDW", "FOXA", "PPL", "LYB", "EL",
    "ROL", "AWK", "FOX", "HBAN", "VLTO", "NTAP", "TROW", "WDC", "DECK", "ATO",
    "FE", "WRB", "TDY", "DVN", "HAL", "ES", "DRI", "LII", "RF", "SBAC",
    "ADM", "NVR", "WAT", "IFF", "ON", "NRG", "CNP", "PHM", "NTRS", "VRSN",
    "STE", "WY", "STX", "CINF", "CBOE", "HUBB", "STLD", "MKC", "PTC", "CMS",
    "LH", "CFG", "TSN", "ERIE", "CTRA", "BIIB", "ZBH", "PODD", "KEY", "BBY",
    "EIX", "PFG", "INVH", "ESS", "PKG", "MAA", "DGX", "NI", "JBL", "TER",
    "TRMB", "CLX", "TPR", "NWS", "LUV", "FFIV", "SNA", "BLDR", "L", "COO",
    "RL", "GPC", "LDOS", "FDS", "NWSA", "FSLR", "JBHT", "GEN", "ULTA", "MAS",
    "DPZ", "ARE", "ZBRA", "OMC", "PNR", "DG", "EXPD", "J", "LNT", "BAX",
    "UDR", "HRL", "WST", "ALGN", "DLTR", "APTV", "EVRG", "MOH", "EPAM", "AKAM",
    "BALL", "KIM", "IEX", "BF.B", "AVY", "AMCR", "CF", "EG", "HOLX", "DOC",
    "KMX", "RVTY", "INCY", "SWK", "TXT", "REG", "POOL", "CPT", "VTRS", "SOLV",
    "MRNA", "DVA", "BXP", "NDSN", "TAP", "JKHY", "JNPR", "CAG", "UHS", "CHRW",
    "MGM", "EMN", "NCLH", "ALLE", "CPB", "HST", "PAYC", "SJM", "BEN", "SWKS",
    "DAY", "TECH", "AIZ", "GL", "LKQ", "PNW", "IPG", "BG", "ALB", "AOS",
    "HSIC", "WYNN", "FRT", "GNRC", "MTCH", "APA", "HAS", "ENPH", "CZR",
    "MOS", "WBA", "LW", "IVZ", "TFX", "CRL", "MHK", "PARA", "CE", "MKTX",
    "AES", "BWA", "HII", "FMC"
]

API_KEY = "K4W4DXDQ5E18FS9X"
DATA_FOLDER = "data_list"  # Folder to store individual stock data
LAST_FETCH_FILE = "last_fetch_date.json"
DAILY_LIMIT = 25  # Maximum requests per day
MAX_RETRIES = 3  # Number of retries for failed requests

# Ensure the data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def load_last_fetch_date():
    """Load the last fetch date to track daily API limit usage."""
    if os.path.exists(LAST_FETCH_FILE):
        with open(LAST_FETCH_FILE, "r") as file:
            return json.load(file)
    return {"last_fetch": None, "count": 0}

def save_last_fetch_date(fetch_count):
    """Save the last fetch date and count."""
    data = {"last_fetch": str(datetime.date.today()), "count": fetch_count}
    with open(LAST_FETCH_FILE, "w") as file:
        json.dump(data, file, indent=4)

def get_historical_eps(stock_symbol, api_key):
    """Fetch historical EPS data for a given stock with retries."""
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={stock_symbol}&apikey={api_key}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)  # 10s timeout
            response.raise_for_status()  # Raise HTTP errors

            data = response.json()

            # Detect API Key errors or rate limits
            if "Information" in data or "Error Message" in data:
                print(f"🚨 API error: {data.get('Information', data.get('Error Message'))}")
                print("❌ Stopping script. Check API key or rate limits.")
                exit(1)  # Stop execution immediately

            if 'annualEarnings' not in data:
                return None

            eps_values = {year['fiscalDateEnding']: float(year['reportedEPS']) for year in data['annualEarnings'] if 'reportedEPS' in year}
            return eps_values

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request error for {stock_symbol}: {e} (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(5)  # Wait before retrying

    print(f"❌ Failed to fetch EPS data for {stock_symbol} after {MAX_RETRIES} attempts.")
    return None  # Return None after max retries

def fetch_eps_data():
    """Fetch EPS data for all S&P 500 stocks with a daily limit of 25 requests and save to individual files."""
    fetch_data = load_last_fetch_date()
    today = str(datetime.date.today())

    # Reset count if a new day starts
    if fetch_data["last_fetch"] != today:
        fetch_data["count"] = 0

    request_count = fetch_data["count"]

    for idx, ticker in enumerate(SP500_TICKERS):
        file_path = os.path.join(DATA_FOLDER, f"{ticker}.json")

        # Skip if already fetched
        if os.path.exists(file_path):
            print(f"✅ {ticker} already processed, skipping...")
            continue
        
        if request_count >= DAILY_LIMIT:
            print(f"🚨 Daily limit of {DAILY_LIMIT} reached. Stopping script for today.")
            break

        print(f"📊 Fetching EPS for {ticker} ({idx+1}/{len(SP500_TICKERS)})...")

        eps_values = get_historical_eps(ticker, API_KEY)
        if eps_values:
            with open(file_path, "w") as file:
                json.dump(eps_values, file, indent=4)
            print(f"💾 EPS data saved for {ticker}")
        else:
            print(f"⚠️ No data for {ticker}")

        request_count += 1
        save_last_fetch_date(request_count)

        # Rate limiting: wait only if needed
        if idx < len(SP500_TICKERS) - 1 and request_count < DAILY_LIMIT:
            time.sleep(12)  # Only wait before the next request

    print(f"✅ Data fetching completed for today. Requests used: {request_count}/{DAILY_LIMIT}")

if __name__ == "__main__":
    fetch_eps_data()