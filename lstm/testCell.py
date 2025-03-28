import numpy as np
import matplotlib.pyplot as plt
import json
from lstm.LSTM_cell import *

class LSTM_test:
    def __init__(self, file_name, n_neurons, n_epoch, plot_each, dt, momentum, decay, learning_rate, auto_skip, price):
        self.file_name = file_name
        self.n_neurons = n_neurons
        self.n_epoch = n_epoch
        self.plot_each = plot_each
        self.dt = dt
        self.momentum = momentum
        self.decay = decay
        self.learning_rate = learning_rate
        self.auto_skip = auto_skip
        self.price = price
        self.intrinsic_value = None
        self.mse = None
        self.scaler_mean = None
        self.scaler_std = None

    def standardize(self, data):
        self.scaler_mean = np.mean(data)
        self.scaler_std = np.std(data)
        return (data - self.scaler_mean) / self.scaler_std

    def unstandardize(self, data):
        return data * self.scaler_std + self.scaler_mean

    def run(self):
        np.random.seed(31)
        random.seed(31)
        with open(self.file_name, 'r') as f:
            data = json.load(f)

        Y_t = np.array(list(data.values()))[::-1]
        Y_t = Y_t.reshape(len(Y_t), 1)
        X_t = np.arange(len(Y_t)).reshape(len(Y_t), 1)

        # Standardize the data
        Y_t_scaled = self.standardize(Y_t)

        [lstm, dense1, dense2, mse, Y_hat_hist] = RunMyLSTM(Y_t_scaled, Y_t_scaled, 
                                    n_neurons = self.n_neurons,\
                                    n_epoch=self.n_epoch, \
                                    plot_each=self.plot_each,\
                                    dt=self.dt,\
                                    momentum=self.momentum, \
                                    decay=self.decay,\
                                    learning_rate=self.learning_rate,\
                                    auto_skip=self.auto_skip)

        self.mse = float(mse)  # Convert MSE to float
        self.rmse = np.sqrt(self.mse)

        Y_hat_scaled = ApplyMyLSTM(Y_t_scaled, lstm, dense1, dense2)
        # Unstandardize the predictions
        Y_hat = self.unstandardize(Y_hat_scaled)
            
        X_plot     = np.arange(0,len(Y_t))
        X_plot_hat = np.arange(0,len(Y_hat)) + self.dt

        self.X_plot_hat = X_plot_hat
        self.Y_hat = Y_hat
        Y_hat_last = Y_hat[-1][0]

        plt.figure(figsize=(12, 6))
        plt.plot(X_plot, Y_t, 'b-', linewidth=2, label='Historical Data')
        plt.plot(X_plot_hat, Y_hat, 'g-', linewidth=2, label='LSTM Prediction')
        plt.axvline(x=len(Y_t)-1, color='k', linestyle='--', label='Prediction Start')
        plt.legend(['y', '$\hat{y}$'])
        plt.show()
    

        # Calculate metrics
        try:
            if self.price is True:  # Added missing colon
                self.growth_percent = 0.0
                self.intrinsic_value = float(Y_hat_last)
            else:
                # Calculate growth rate using the ratio of last predicted value to last actual value
                self.growth_percent = float(((Y_hat[-1] / Y_t[-1]) ** (1/self.dt) - 1) * 100)
                current_eps = float(Y_t[-1])  # Use the last actual value
                bond_yield = 4.8
                self.intrinsic_value = float(current_eps * (7.5 + 1 * self.growth_percent) * (4.4/bond_yield))
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            self.growth_percent = 0.0
            self.intrinsic_value = 0.0

        print(f"\nGrowth Rate: {self.growth_percent:.2f}%")
        print(f"Intrinsic Value using Benjamin Graham's Formula (LSTM): ${self.intrinsic_value:.2f}")