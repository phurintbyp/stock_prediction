import requests
import json
import os
from datetime import datetime

# Replace with your Alpha Vantage API key
API_KEY = "VWZMI0HMRGA407QK"
BASE_URL = "https://www.alphavantage.co/query"
OUTPUT_DIR = "e:/stock_prediction/fcf"

def fetch_fcf_data(symbol):
    params = {
        "function": "CASH_FLOW",
        "symbol": symbol,
        "apikey": API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        annual_reports = data.get('annualReports', [])
        
        # Create dictionary with date keys and FCF values
        fcf_dict = {}
        for report in annual_reports:
            date = report['fiscalDateEnding']
            operating_cashflow = float(report.get('operatingCashflow', 0))
            capital_expenditure = float(report.get('capitalExpenditures', 0))
            fcf = int(operating_cashflow - capital_expenditure)  # Convert to integer
            fcf_dict[date] = fcf
        
        return fcf_dict
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

def save_to_json(symbol, data):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{symbol}_fcf_{timestamp}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)  # Added sort_keys=True to maintain date order
    print(f"Data saved to {filepath}")

def main(symbols):
    for symbol in symbols:
        try:
            data = fetch_fcf_data(symbol)
            save_to_json(symbol, data)
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

if __name__ == "__main__":
    # Add your stock symbols here
    stock_symbols = ["GOOGL"]
    main(stock_symbols)
