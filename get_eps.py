import requests
import json

def get_historical_eps(stock_symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={stock_symbol}&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return 'Error fetching data'
    
    data = response.json()
    if 'annualEarnings' not in data:
        return 'No data available'
    
    eps_values = {year['fiscalDateEnding']: float(year['reportedEPS']) for year in data['annualEarnings'] if 'reportedEPS' in year}
    return eps_values

def save_eps_to_file(eps_values, filename="eps_data.json"):
    with open(filename, "w") as file:
        json.dump(eps_values, file, indent=4)

if __name__ == "__main__":
    stock_symbol = "AAPL"
    api_key = "Z0IXDUZQ5UQTWXK9"
    num = "2"
    eps_values = get_historical_eps(stock_symbol, api_key)
    
    if isinstance(eps_values, str):
        print(eps_values)
    else:
        print(f"The historical EPS values of {stock_symbol} are:")
        for year, value in eps_values.items():
            print(f"{year}: {value}")
        
        save_eps_to_file(eps_values, f"./data/eps_data_{num}.json")
        print("EPS data has been saved to eps_data.json")
