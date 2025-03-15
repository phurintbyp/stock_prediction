import numpy as np
import matplotlib.pyplot as plt
import json
from RNN.MyRNN import *


class RNN_test:
    
    def __init__(self, file_name, n_epoch, n_neurons, learning_rate, decay, momentum, dt, auto_skip, price):
        self.file_name = file_name
        self.n_epoch = n_epoch
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.dt = dt
        self.auto_skip = auto_skip
        self.price = price
        self.intrinsic_value = None
        self.mse = None
        self.mean = None
        self.std = None

    def standardize(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean) / self.std

    def unstandardize(self, data):
        return (data * self.std) + self.mean

    def run(self):
        np.random.seed(31)

        with open(self.file_name, 'r') as f:
            data = json.load(f)

        dates = np.array(list(data.keys()))[::-1]
        values = np.array(list(data.values()))[::-1]

        X_t = np.arange(len(dates)).reshape(len(dates), -1)
        Y_t = values.reshape(len(values), 1)

        Y_t_standardized = self.standardize(Y_t)

        [rnn, mse] = RunMyRNN(
            X_t,
            Y_t_standardized,
            Activation=Tanh(),
            n_epoch=self.n_epoch,         # can increase more if needed
            n_neurons=self.n_neurons,        # fewer neurons for small dataset
            learning_rate=self.learning_rate,  # higher LR to avoid slow/stuck training
            decay=self.decay,          # tiny weight decay
            momentum=self.momentum,        
            dt=self.dt,
            auto_skip=self.auto_skip
        )
        self.mse = float(mse)

        Y_hat_standardized = ApplyMyRNN(Y_t_standardized, rnn)
        Y_hat = self.unstandardize(Y_hat_standardized)

        X_t   = np.arange(len(Y_t))
        X_full = X_t + self.dt
        self.X_full = X_full
        self.Y_hat = Y_hat

        plt.figure(figsize=(12, 6))
        plt.plot(X_t, Y_t, 'b-', linewidth=2, label='Historical Data')
        plt.plot(X_full, Y_hat, 'g-', linewidth=2, label='LSTM Prediction')
        plt.axvline(x=len(Y_t)-1, color='k', linestyle='--', label='Prediction Start')
        plt.legend(['y', '$\hat{y}$'])
        plt.show()

        Y_t_last = Y_t[-1][0]
        Y_hat_last = Y_hat[-1][0]
        
        try:
            if self.price is True:  # Added missing colon
                self.growth_percent = 0.0
                self.intrinsic_value = float(Y_hat_last)
            else:
                growth_percent = ((Y_hat_last / Y_t_last) ** (1 / self.dt) - 1) * 100
                bond_yield = 4.8
                current_eps = Y_t_last
                
                self.growth_percent = growth_percent
                self.intrinsic_value = current_eps * (7.5 + 1 * growth_percent) * (4.4/bond_yield)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            self.growth_percent = 0.0
            self.intrinsic_value = 0.0
            
        print("Intrinsic Value using Benjamin Graham's Formula(RNN):", self.intrinsic_value)
