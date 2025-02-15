import numpy as np
import json
import matplotlib.pyplot as plt
import datetime

file_name = "./data/eps_data.json"
year_interval = 1

def calculate_g(file_name, year_interval): 
    with open(file_name, 'r') as file:
        data = json.load(file)

    dates = list(reversed(data.keys()))
    values = list(reversed(data.values()))

    g_list = []

    for i in range(0, (len(values) - 1)):
        g = (((values[i+1] / values[i])**(1/year_interval)) - 1)
        g_list.append((round(g, 5), dates[i+1]))

    return g_list

if __name__ == "__main__":
    g = calculate_g(file_name, year_interval)
    y_values = [point[0] for point in g]
    x_dates = [datetime.datetime.strptime(point[1], '%Y-%m-%d') for point in g]

    plt.figure(figsize=(12, 6))
    plt.plot(x_dates, y_values, marker="o", linestyle="-", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title("Growth Rate data")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    print(g)