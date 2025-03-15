import numpy as np
import matplotlib.pyplot as plt
import json
from lstm.LSTM_cell import *

class LSTM_test:
    def __init__(self, file_name, n_neurons, n_epoch, plot_each, dt, momentum, decay, learning_rate, auto_skip):
        self.file_name = file_name
        self.n_neurons = n_neurons
        self.n_epoch = n_epoch
        self.plot_each = plot_each
        self.dt = dt
        self.momentum = momentum
        self.decay = decay
        self.learning_rate = learning_rate
        self.auto_skip = auto_skip
        self.intrinsic_value = None
        self.mse = None

    def run(self):
        with open(self.file_name, 'r') as f:
            data = json.load(f)

        Y_t = np.array(list(data.values()))[::-1]
        Y_t = Y_t.reshape(len(Y_t), 1)
        X_t = np.arange(len(Y_t)).reshape(len(Y_t), 1)

        [lstm, dense1, dense2, mse, Y_hat_hist] = RunMyLSTM(Y_t, Y_t, 
                                    n_neurons = self.n_neurons,\
                                    n_epoch=self.n_epoch, \
                                    plot_each=self.plot_each,\
                                    dt=self.dt,\
                                    momentum=self.momentum, \
                                    decay=self.decay,\
                                    learning_rate=self.learning_rate,\
                                    auto_skip=self.auto_skip)

        self.mse = float(mse)  # Convert MSE to float

        Y_hat     = ApplyMyLSTM(Y_t,lstm, dense1, dense2)
            
        X_plot     = np.arange(0,len(Y_t))
        X_plot_hat = np.arange(0,len(Y_hat)) + self.dt

        plt.figure(figsize=(12, 6))
        plt.plot(X_plot, Y_t, 'b-', linewidth=2, label='Historical Data')
        plt.plot(X_plot_hat, Y_hat, 'g-', linewidth=2, label='LSTM Prediction')
        plt.axvline(x=len(Y_t)-1, color='k', linestyle='--', label='Prediction Start')
        plt.legend(['y', '$\hat{y}$'])
        plt.show()
    

        # Calculate metrics
        try:
            # Calculate growth rate using the ratio of last predicted value to last actual value
            # Using Y_hat[1:] since Y_hat includes one extra prediction point
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