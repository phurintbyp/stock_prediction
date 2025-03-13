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
        mean_y = np.mean(Y_t)
        std_y = np.std(Y_t)
        Y_t_scaled = (Y_t - mean_y) / std_y

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
            Y_t_scaled,          # <--- use scaled target here
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

        # 7) Predict next 10 steps using the trained model
        future_steps = 10
        X_new = np.arange(len(X_t), len(X_t) + future_steps).reshape(-1, 1)

        # 8) Get scaled predictions
        Y_new_scaled = ApplyMyRNN(X_new, rnn)

        # 9) Invert the scaling for final predictions: Y_new = Y_scaled * std_y + mean_y
        Y_new = (Y_new_scaled * std_y) + mean_y
        print(f"Predictions: {Y_new}")

        # 10) Plot historical vs. predicted
        # plt.figure(figsize=(10, 6))
        # plt.plot(X_t, Y_t, 'b-', label='Historical Data')
        # plt.plot(X_new, Y_new, 'r--', label='10-Year Prediction')
        # plt.title("Value Prediction (Scaled RNN)")
        # plt.xlabel("Time Index")
        # plt.ylabel("Value")
        # plt.legend()

        # # Add vertical line to show where prediction starts
        # pred_start = len(X_t) - 1
        # plt.axvline(x=pred_start, color='k', linestyle=':', alpha=0.5)
        # plt.text(pred_start + 0.5, np.min(Y_t),
        #          'Prediction Start', rotation=90)

        # fig = plt.gcf()
        # fig.canvas.mpl_connect('key_press_event', close_on_key)
        # plt.show()

        # 11) Print out the future predictions
        final_historical_year = 2024
        prediction_years = [final_historical_year + i + 1 for i in range(future_steps)]
        print("Predicted values for future years:")
        for year, value in zip(prediction_years, Y_new.flatten()):
            print(f"{year}: {value:.2f}")

        # Invert scaling for the historical predictions (stored in rnn.Y_hat) to get back to the original scale.
        Y_hat_hist = (rnn.Y_hat * std_y) + mean_y  # model’s prediction on historical data

        # Y_new has been inverted already (or invert it similarly if needed)
        # Y_new = (Y_new_scaled * std_y) + mean_y   # if not already inverted

        # Concatenate the historical predictions with future predictions
        Y_hat_full = np.concatenate([Y_hat_hist, Y_new])

        # Create a full x-axis that spans the historical period and the future predictions
        X_full = np.arange(0, len(X_t) + future_steps).reshape(-1, 1)
        self.X_full = X_full
        self.Y_hat_full = Y_hat_full

        plt.figure(figsize=(12, 6))
        # Plot the model's prediction over the full period (historical + future)
        plt.plot(X_full, Y_hat_full, 'r-', linewidth=2, label='Model Prediction')
        # Plot the actual historical data (only for the historical period)
        plt.plot(X_t, Y_t, 'b-', linewidth=2, label='Historical Data')

        # Mark the transition from historical to predicted future data
        plt.axvline(x=len(X_t) - 1, color='k', linestyle='--', label='Prediction Start')

        plt.title("Historical Data vs. Full Model Prediction")
        plt.xlabel("Time Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        growth_percent = ((Y_new[-1][0] / Y_t[-1][0]) ** (1 / 10) - 1) * 100
        bond_yield = 4.8
        current_eps = Y_t[-1][0]
        self.growth_percent = growth_percent
        self.intrinsic_value = current_eps * (7.5 + 1 * growth_percent) * (4.4/bond_yield)
        print("Intrinsic Value using Benjamin Graham's Formula(LSTM):", self.intrinsic_value)
