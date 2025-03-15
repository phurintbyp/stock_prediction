import requests
import json
import os
import time

# Configuration
API_KEY = "VWZMI0HMRGA407QK"  # Replace with your Alpha Vantage API Key
STOCK_SYMBOL = "NFLX"  # Replace with your desired stock symbol
DATA_FOLDER = "./data/monthly_prices"  # Folder to store monthly stock price data
MAX_RETRIES = 3  # Number of retries for failed requests
SLEEP_TIME = 12  # Sleep time between API requests (to avoid rate limits)

# Ensure the data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def fetch_monthly_prices(stock_symbol, api_key):
    """Fetch monthly adjusted stock price data from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={stock_symbol}&apikey={api_key}"
    
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

            # Extract monthly adjusted closing prices
            if "Monthly Adjusted Time Series" not in data:
                print(f"⚠️ No monthly data found for {stock_symbol}")
                return None

            monthly_prices = {
                date: float(values["5. adjusted close"])
                for date, values in data["Monthly Adjusted Time Series"].items()
            }

            return monthly_prices

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request error for {stock_symbol}: {e} (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(SLEEP_TIME)  # Wait before retrying
    
    print(f"❌ Failed to fetch monthly price data for {stock_symbol} after {MAX_RETRIES} attempts.")
    return None

def save_price_data(stock_symbol, price_data):
    """Save monthly stock price data to a JSON file."""
    file_path = os.path.join(DATA_FOLDER, f"{stock_symbol}.json")
    with open(file_path, "w") as file:
        json.dump(price_data, file, indent=4)
    print(f"💾 Monthly stock price data saved for {stock_symbol} in {file_path}")

def main():
    print(f"📈 Fetching monthly stock prices for {STOCK_SYMBOL}...")

    price_data = fetch_monthly_prices(STOCK_SYMBOL, API_KEY)
    
    if price_data:
        save_price_data(STOCK_SYMBOL, price_data)
        print("📊 Monthly Prices:")
        for date, price in sorted(price_data.items(), reverse=True):
            print(f"{date}: {price:.2f}")
    else:
        print(f"⚠️ No monthly price data available for {STOCK_SYMBOL}")

    print("✅ Done.")

if __name__ == "__main__":
    main()
