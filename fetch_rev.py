import requests
import json
import os
import datetime
import time

# Configuration
API_KEY = "VWZMI0HMRGA407QK"  # Replace with your Alpha Vantage API Key
STOCK_SYMBOL = "GOOGL"  # Replace with your desired stock symbol
DATA_FOLDER = "revenue"  # Folder to store revenue data
MAX_RETRIES = 3  # Number of retries for failed requests
SLEEP_TIME = 12  # Sleep time between API requests (to avoid rate limits)

# Ensure the "revenue" folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def fetch_revenue(stock_symbol, api_key):
    """Fetch revenue data from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={stock_symbol}&apikey={api_key}"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)  # 10s timeout
            response.raise_for_status()  # Raise HTTP errors
            data = response.json()
            
            # Check for errors in API response
            if "Information" in data or "Error Message" in data:
                print(f"🚨 API error: {data.get('Information', data.get('Error Message'))}")
                print("❌ Stopping script. Check API key or rate limits.")
                return None

            # Extract revenue data
            if "annualReports" not in data:
                print(f"⚠️ No revenue data found for {stock_symbol}")
                return None

            revenue_data = {
                report["fiscalDateEnding"]: report["totalRevenue"]
                for report in data["annualReports"]
                if "totalRevenue" in report
            }

            return revenue_data

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request error for {stock_symbol}: {e} (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(SLEEP_TIME)  # Wait before retrying
    
    print(f"❌ Failed to fetch revenue data for {stock_symbol} after {MAX_RETRIES} attempts.")
    return None

def save_revenue_data(stock_symbol, revenue_data):
    """Save revenue data to a JSON file."""
    file_path = os.path.join(DATA_FOLDER, f"{stock_symbol}.json")
    with open(file_path, "w") as file:
        json.dump(revenue_data, file, indent=4)
    print(f"💾 Revenue data saved for {stock_symbol} in {file_path}")

def main():
    print(f"📊 Fetching revenue for {STOCK_SYMBOL}...")

    revenue_data = fetch_revenue(STOCK_SYMBOL, API_KEY)
    
    if revenue_data:
        save_revenue_data(STOCK_SYMBOL, revenue_data)
    else:
        print(f"⚠️ No revenue data available for {STOCK_SYMBOL}")

    print("✅ Done.")

if __name__ == "__main__":
    main()