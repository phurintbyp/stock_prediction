import requests
import json
import os
import time

# Configuration
API_KEY = "VWZMI0HMRGA407QK"  # Replace with your Alpha Vantage API Key
STOCK_SYMBOL = ""  # Replace with your desired stock symbol
DATA_FOLDER = "./data/daily_prices"  # Folder to store daily stock price data
MAX_RETRIES = 3
SLEEP_TIME = 12  # Avoid API rate limit

# Ensure the data folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def fetch_daily_prices(stock_symbol, api_key):
    """Fetch all daily stock price data from Alpha Vantage (non-adjusted)."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&apikey={api_key}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "Information" in data or "Error Message" in data:
                print(f"🚨 API error: {data.get('Information', data.get('Error Message'))}")
                return None

            if "Time Series (Daily)" not in data:
                print(f"⚠️ No daily data found for {stock_symbol}")
                return None

            daily_prices = {
                date: float(values["4. close"])
                for date, values in data["Time Series (Daily)"].items()
            }

            return daily_prices

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request error for {stock_symbol}: {e} (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(SLEEP_TIME)

    print(f"❌ Failed to fetch daily price data for {stock_symbol} after {MAX_RETRIES} attempts.")
    return None

def save_price_data(stock_symbol, price_data):
    """Save daily stock price data to a JSON file."""
    file_path = os.path.join(DATA_FOLDER, f"{stock_symbol}.json")
    with open(file_path, "w") as file:
        json.dump(price_data, file, indent=4)
    print(f"💾 Daily stock price data saved for {stock_symbol} in {file_path}")

def main():
    print(f"📈 Fetching daily stock prices for {STOCK_SYMBOL}...")

    price_data = fetch_daily_prices(STOCK_SYMBOL, API_KEY)

    if price_data:
        save_price_data(STOCK_SYMBOL, price_data)
        print("📊 Daily Prices:")
        for date, price in sorted(price_data.items(), reverse=True)[:10]:  # Show top 10 most recent
            print(f"{date}: {price:.2f}")
    else:
        print(f"⚠️ No daily price data available for {STOCK_SYMBOL}")

    print("✅ Done.")

if __name__ == "__main__":
    main()