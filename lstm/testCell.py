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

        # Plot the original data
        plt.plot(Y_t)
        plt.title('Earnings Per Share')
        plt.show()

        [lstm, dense1, dense2, self.mse] = RunMyLSTM(Y_t, Y_t, n_neurons=self.n_neurons,
                                        n_epoch=self.n_epoch, plot_each=self.plot_each, dt=self.dt,
                                        momentum=self.momentum, decay=self.decay,
                                        learning_rate=self.learning_rate,
                                        auto_skip=self.auto_skip)  # Add auto_skip parameter

        Y_hat = ApplyMyLSTM(Y_t, lstm, dense1, dense2)
            
        X_plot = np.arange(0, len(Y_t))
        X_plot_hat = np.arange(0, len(Y_hat)) + self.dt

        plt.plot(X_plot, Y_t)
        plt.plot(X_plot_hat, Y_hat)
        plt.legend(['Actual EPS', 'Predicted EPS'])
        plt.title('Earnings Prediction')
        plt.show()

        growth_percent = ((Y_hat[-1] / Y_t[-1]) ** (1/self.dt) - 1) * 100
        current_eps = Y_t[-1][0]
        bond_yield = 4.8

        self.intrinsic_value = current_eps * (7.5 + 1 * growth_percent) * (4.4/bond_yield)