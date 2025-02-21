import numpy as np
import matplotlib.pyplot as plt
import json
from utils.linear_regression import Linear_regression
from utils.calculate_data import calculate_g
import datetime

file_name = "./data_list/ABT.json"
degree = 4
year_interval = 10

with open(file_name, 'r') as file:
    data = json.load(file)

dates = list(reversed(data.keys()))
values = list(reversed(data.values()))

eps_list = [(int(dates[i][:4]), values[i]) for i in range(len(values))]

year = np.array([item[0] for item in eps_list]).reshape(-1, 1)
eps = np.array([item[1] for item in eps_list])

lr = Linear_regression()
lr.fit(year, eps, degree)

last_year = year[-1][0]
future_years = np.arange(last_year + 1, last_year + 11).reshape(-1, 1)

extended_years = np.vstack((year, future_years))
predicted_eps = lr.predict(extended_years, degree)

last_ten_years = eps[-10:].tolist()
predicted_eps_rounded = np.round(predicted_eps[len(year):], 2).tolist()

combined_eps_dict = {year[i][0]: eps[i] for i in range(len(year))}
combined_eps_dict.update({future_years[i][0]: predicted_eps_rounded[i][0] for i in range(len(future_years))})

predicted_value_10_years = predicted_eps[-1]
print(f"Predicted EPS in {last_year + 10}: {predicted_value_10_years.item():.4f}")
print(f"Last 10 years of EPS data: {last_ten_years}")
print(f"Combined EPS dictionary: {combined_eps_dict}")

plt.figure(figsize=(12, 6))
plt.scatter(year, eps, color="orange", label="EPS Data")
plt.plot(extended_years[:len(year)], predicted_eps[:len(year)], marker="o", linestyle="-", linewidth=2, color="blue", label="Predicted Data")
plt.plot(extended_years[len(year)-1:], predicted_eps[len(year)-1:], marker="o", linestyle="-", linewidth=2, color="green", label="Predicted Data")
plt.xlabel("Year")
plt.ylabel("Value")
plt.title("Growth Rate Data (Extended for 10 Years)")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.show()

g = calculate_g(combined_eps_dict, year_interval)
y_values = [point[0] for point in g]
x_dates = [datetime.datetime.strptime(str(point[1]), '%Y') for point in g]

split_index = len(y_values) - 10
x_actual = x_dates[:split_index]
y_actual = y_values[:split_index]
x_predicted = x_dates[split_index-1:]
y_predicted = y_values[split_index-1:]

plt.figure(figsize=(12, 6))
plt.plot(x_actual, y_actual, marker="o", linestyle="-", linewidth=2, color="green", label="Actual CAGR")
plt.plot(x_predicted, y_predicted, marker="o", linestyle="-", linewidth=2, color="orange", label="Predicted CAGR")
plt.xlabel("Year")
plt.ylabel("CAGR Value")
plt.title(f"CAGR Data Over {year_interval}-Year Intervals")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.show()