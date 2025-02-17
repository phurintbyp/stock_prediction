import numpy as np
import matplotlib.pyplot as plt
import json
from linear_regression import Linear_regression

# Load EPS data
file_name = "./data/eps_data_2.json"
degree = 4

with open(file_name, 'r') as file:
    data = json.load(file)

# Extract and reverse data for chronological order
dates = list(reversed(data.keys()))
values = list(reversed(data.values()))

# Convert to numerical year-based format
eps_list = [(i + 1, values[i]) for i in range(len(values))]

# Convert to NumPy arrays
year = np.array([item[0] for item in eps_list]).reshape(-1, 1)  # Reshape for sklearn compatibility
eps = np.array([item[1] for item in eps_list])

# Train Linear Regression model
lr = Linear_regression()
lr.fit(year, eps, degree)

# Extend for 10 more years beyond the last year
last_year = year[-1][0]
future_years = np.arange(last_year + 1, last_year + 11).reshape(-1, 1)

extended_years = np.vstack((year, future_years))
predicted_eps = lr.predict(extended_years, degree)

plt.figure(figsize=(12, 6))
plt.scatter(year, eps, label="Actual Data", color="blue")
plt.plot(extended_years, predicted_eps, marker="o", linestyle="-", linewidth=2, color="red", label="Predicted Growth")
plt.xlabel("Year")
plt.ylabel("Value")
plt.title("Growth Rate Data (Extended for 10 Years)")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.show()
