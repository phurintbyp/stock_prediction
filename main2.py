import numpy as np
import matplotlib.pyplot as plt
import json
from utils.linear_regression import Linear_regression

class EPSPrediction:
    def __init__(self, file_name, degree, bond_yield):
        self.file_name = file_name
        self.degree = degree
        self.year = None
        self.eps = None
        self.extended_years = None
        self.predicted_eps = None
        self.bond_yield = bond_yield
        self.Y_true = 0

    def load_data(self):
        with open(self.file_name, 'r') as file:
            data = json.load(file)
        
        dates = list(reversed(data.keys()))
        values = list(reversed(data.values()))
        eps_list = [(i + 1, values[i]) for i in range(len(values))]
        
        self.year = np.array([item[0] for item in eps_list]).reshape(-1, 1)
        self.eps = np.array([item[1] for item in eps_list])

    def regression(self):
        lr = Linear_regression()
        lr.fit(self.year, self.eps, self.degree)
        last_year = self.year[-1][0]
        future_years = np.arange(last_year + 1, last_year + 21).reshape(-1, 1)  # Extend for 20 more years
        self.extended_years = np.vstack((self.year, future_years))
        self.predicted_eps = lr.predict(self.extended_years, self.degree)

    def plot_data(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(self.year, self.eps, label="Actual Data", color="blue")
        plt.plot(self.extended_years, self.predicted_eps, marker="o", linestyle="-", linewidth=2, color="red", label="Predicted Growth")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.title("Growth Rate Data (Extended for 20 Years)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
    
    def printvalue(self):
        current_eps = self.eps[-1]
        desired_year = self.year[-1][0] + 10
        index = desired_year - 1  # Convert to 0-based index
        eps_10_years = self.predicted_eps[index][0]
        growth = (eps_10_years / current_eps) ** (1/10) - 1
        growth_percent = growth * 100  # Convert to percentage for formula
        
        # Revised intrinsic value formula with 4.4 adjustment
        intrinsic_value = current_eps * (7 + 1 * growth_percent) * (4.4/bond_yield)
        
        print("Intrinsic Value using Benjamin Graham's Formula:", intrinsic_value)
        print("Current EPS:", current_eps)
        print("Future EPS 10 years from now:", eps_10_years)
        print("Predicted Growth Rate (%):", growth_percent)

    def run(self):
        self.load_data()
        self.regression()
        self.plot_data()
        self.printvalue()
    
    def MSE(self):
        self.Y_true = 0  # Original Value
        for i in range(len(self.eps)):
            loss = (self.eps[i] - self.predicted_eps[i]) ** 2
            self.Y_true += loss
        self.MSE = self.Y_true / len(self.eps)
        return self.MSE

# Example usage
file_name = "./data_list/AAPL.json"
degree = 4
bond_yield = 4.8
eps_prediction = EPSPrediction(file_name, degree, bond_yield)
eps_prediction.run()
print("MSE: ", eps_prediction.MSE())