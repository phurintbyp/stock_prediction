import numpy as np
import json
import matplotlib.pyplot as plt
import datetime

file_name = "./data_list/ABT.json"
year_interval = 10

def calculate_g(combined_eps_dict, year_interval):
    years = sorted(combined_eps_dict.keys())  # Sort years to ensure chronological order
    g_list = []
    
    for i in range(len(years) - year_interval):
        try:
            start_year = years[i]
            end_year = years[i + year_interval]
            start_value = combined_eps_dict[start_year]
            end_value = combined_eps_dict[end_year]
            
            g = (((end_value / start_value) ** (1 / year_interval)) - 1)
            g_list.append((round(g, 5), end_year))
        except (IndexError, KeyError, ZeroDivisionError):
            pass
    
    return g_list

if __name__ == "__main__":
    # Load the combined EPS dictionary from the modified script
    with open("./data_list/combined_eps.json", "r") as file:
        combined_eps_dict = json.load(file)
    
    g = calculate_g(combined_eps_dict, year_interval)
    y_values = [point[0] for point in g]
    x_dates = [datetime.datetime.strptime(str(point[1]), '%Y') for point in g]  # Convert year to datetime format

    plt.figure(figsize=(12, 6))
    plt.plot(x_dates, y_values, marker="o", linestyle="-", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("CAGR Value")
    plt.title("Growth Rate Data (CAGR over {} years)".format(year_interval))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()