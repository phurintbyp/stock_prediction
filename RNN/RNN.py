import numpy as np
import matplotlib.pyplot as plt
import json
from RNN.MyRNN import *


class RNN_test:
    
    def __init__(self, file_name, n_epoch, n_neurons, learning_rate, decay, momentum, dt, auto_skip):
        self.file_name = file_name
        self.n_epoch = n_epoch
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.dt = dt
        self.auto_skip = auto_skip
        self.intrinsic_value = None
        self.mse = None

    def run(self):
        np.random.seed(31)

        # 1) Load data
        with open(self.file_name, 'r') as f:
            data = json.load(f)

        # Reverse dictionaries for chronological order (earliest -> latest)
        dates = np.array(list(data.keys()))[::-1]
        values = np.array(list(data.values()))[::-1]

        # 2) Prepare data: X_t is just an index; Y_t is the (original) target
        X_t = np.arange(len(dates)).reshape(len(dates), -1)
        Y_t = values.reshape(len(values), 1)

        # 3) Standardize Y_t: Y_scaled = (Y - mean) / std
        # mean_y = np.mean(Y_t)
        # std_y = np.std(Y_t)
        # Y_t_scaled = (Y_t - mean_y) / std_y

        # 4) Quick check of scaled range (optional printout)
        # print("Scaled range:", Y_t_scaled.min(), Y_t_scaled.max())

        # 5) Visualize original (unscaled) data
        plt.plot(X_t, Y_t, label="Original Data")
        plt.title("Historical Data")
        plt.xlabel("Time Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        # 6) Train RNN on the SCALED Y data
        [rnn, mse] = RunMyRNN(
            X_t,
            Y_t,
            Activation=Tanh(),
            n_epoch=self.n_epoch,         # can increase more if needed
            n_neurons=self.n_neurons,        # fewer neurons for small dataset
            learning_rate=self.learning_rate,  # higher LR to avoid slow/stuck training
            decay=self.decay,          # tiny weight decay
            momentum=self.momentum,        
            dt=self.dt,
            auto_skip=self.auto_skip
        )
        self.mse = float(mse)  # Convert MSE to float

        Y_hat = ApplyMyRNN(Y_t, rnn)

        X_t   = np.arange(len(Y_t))

        plt.figure(figsize=(12, 6))
        plt.plot(X_t, Y_t, 'b-', linewidth=2, label='Historical Data')
        plt.plot(X_t + self.dt, Y_hat, 'g-', linewidth=2, label='LSTM Prediction')
        plt.axvline(x=len(Y_t)-1, color='k', linestyle='--', label='Prediction Start')
        plt.legend(['y', '$\hat{y}$'])
        plt.show()

        # Calculate Metrics
        Y_t_last = Y_t[-1][0]  # Get the last historical value
        Y_hat_last = Y_hat[-1][0]  # Get the last predicted value
        
        # Calculate annual growth rate based on actual vs predicted
        growth_percent = ((Y_hat_last / Y_t_last) ** (1 / self.dt) - 1) * 100
        bond_yield = 4.8
        current_eps = Y_t_last
        
        self.growth_percent = growth_percent
        self.intrinsic_value = current_eps * (7.5 + 1 * growth_percent) * (4.4/bond_yield)
        print("Intrinsic Value using Benjamin Graham's Formula(RNN):", self.intrinsic_value)
