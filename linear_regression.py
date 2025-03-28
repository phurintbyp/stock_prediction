import numpy as np
import matplotlib.pyplot as plt
import json
from utils.linear_regression import Linear_regression

class EPSPrediction:
    def __init__(self, file_name, degree, bond_yield, price, dt):
        self.file_name = file_name
        self.degree = degree
        self.year = None
        self.eps = None
        self.extended_years = None
        self.predicted_eps = None
        self.bond_yield = bond_yield
        self.Y_true = 0
        self.intrinsic_value = None
        self.mse = None
        self.price = "quarterly_prices" in file_name
        self.dt = dt
        self.rmse = None

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
        plt.title("Growth Rate Data")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
    
    def printvalue(self):
        current_eps = self.eps[-1]
        desired_year = self.year[-1][0] + self.dt
        index = desired_year - 1  # Convert to 0-based index
        future_eps = self.predicted_eps[index][0]
        
        try:
            if self.price:
                self.growth_percent = 0.0
                self.intrinsic_value = float(self.predicted_eps[index][0])
            else:
                growth = (future_eps / current_eps) ** (1/self.dt) - 1
                growth_percent = growth * 100  # Convert to percentage for formula
                self.growth_percent = growth_percent
                # Revised intrinsic value formula with 4.4 adjustment
                self.intrinsic_value = current_eps * (7.5 + 1 * growth_percent) * (4.4/self.bond_yield)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            self.growth_percent = 0.0
            self.intrinsic_value = 0.0
            
        self.mse = self.MSE()
        self.rmse = self.RMSE()
        print("Intrinsic Value using Benjamin Graham's Formula:", self.intrinsic_value)
        print("Current EPS:", current_eps)
        print(f"Future EPS {self.dt} years from now:", future_eps)
        print("Predicted Growth Rate (%):", self.growth_percent)
        print("Mean Squared Error:", self.mse)

    def run(self):
        self.load_data()
        self.regression()
        self.plot_data()
        self.printvalue()
        
    def MSE(self):
        self.loss_sum = 0
        for i in range(len(self.eps)):
            if self.price:
                # For prices, use percentage error
                actual = self.eps[i]
                predicted = self.predicted_eps[i]
                percent_error = (actual - predicted) / actual if actual != 0 else 0
                self.loss_sum += percent_error ** 2
            else:
                # For EPS, use regular MSE
                self.loss_sum += (self.eps[i] - self.predicted_eps[i])**2
        
        self.mse = float(self.loss_sum / len(self.eps))
        return self.mse
    
    def RMSE(self):
        self.rmse = np.sqrt(self.mse)
        return self.rmse
    
    def RSQ(self):
        # Get predictions for the known data points only
        Y_hat = self.predicted_eps[:len(self.eps)]
        Y_t = self.eps
        
        # Calculate R-squared
        ss_res = np.sum((Y_t - Y_hat) ** 2)
        ss_tot = np.sum((Y_t - np.mean(Y_t)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
    
if __name__ == "__main__": 
    # Example usage
    file_name = "./data/data_list/USB.json"
    degree = 2
    bond_yield = 4.8
    dt = 10
    eps_prediction = EPSPrediction(file_name, degree, bond_yield, price=False, dt=dt)
    eps_prediction.run()
    print("Mean Squared Error:", eps_prediction.MSE())
    print("R-Squared:", eps_prediction.RSQ())

