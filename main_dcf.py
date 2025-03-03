import numpy as np
import matplotlib.pyplot as plt
import json
from utils.linear_regression import Linear_regression
from dcf import DCF

class RevPrediction:
    def __init__(self, file_name, n_predict_years, degree=1):
        self.file_name = file_name
        self.degree = degree
        self.n_predict_years = n_predict_years
        self.year = None
        self.rev = None
        self.extended_years = None
        self.predicted_rev = None
        self.growth = []

    def load_data(self):
        with open(self.file_name, 'r') as file:
            data = json.load(file)
        
        dates = list(reversed(data.keys()))
        values = list(reversed(data.values()))
        rev_list = [(i + 1, values[i]) for i in range(len(values))]
        
        self.year = np.array([item[0] for item in rev_list]).reshape(-1, 1)
        self.rev = np.array([item[1] for item in rev_list])

    def regression(self):
        lr = Linear_regression()
        lr.fit(self.year, self.rev, self.degree)
        last_year = self.year[-1][0]
        future_years = np.arange(last_year + 1, last_year + self.n_predict_years + 1).reshape(-1, 1)  # Extend for 20 more years
        self.extended_years = np.vstack((self.year, future_years))
        self.predicted_rev = lr.predict(self.extended_years, self.degree)

    def plot_data(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(self.year, self.rev, label="Actual Data", color="blue")
        plt.plot(self.extended_years, self.predicted_rev, marker="o", linestyle="-", linewidth=2, color="red", label="Predicted Growth")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.title("Growth Rate Data (Extended for 20 Years)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(self.extended_years[:len(self.growth)], self.growth, marker="o", linestyle="-", linewidth=2, color="red", label="Predicted Growth")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.title("Growth Rate Data (Extended for 20 Years)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
    
    def printvalue(self):
        n_predictions = len(self.predicted_rev)

        if n_predictions < 10:
            print("Warning: Not enough data to calculate growth.")
            self.growth = []
            return
        
        for i in range(n_predictions - 10):  # Ensures i+10 is within bounds
            self.growth.append((self.predicted_rev[i + 10] / self.predicted_rev[i]) ** (1/10) - 1)
        
        print(str(self.growth[-1] * 100))

    def run(self):
        self.load_data()
        self.regression()
        self.printvalue()
        self.plot_data()

############################################################################
# predict revenue
############################################################################

file_name = "./revenue/GOOGL.json"
n_predicted_years = 20
rev_prediction = RevPrediction(file_name, n_predicted_years)
rev_prediction.run()

############################################################################
# Calculate DCF
############################################################################

# GOOGL data
fcf = [42843, 67012, 60010,	69495, 72764]
wacc = 0.0943
debt = 28140
cash = 95660
shares_outstanding = 12210

growth_rate = rev_prediction.growth
print(growth_rate)

dcf = DCF(fcf, wacc, growth_rate, debt, cash, shares_outstanding)
intrinsic_value = dcf.calculate_intri()
print(intrinsic_value)