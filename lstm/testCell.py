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

        # Standardize data
        mean_y = np.mean(Y_t)
        std_y = np.std(Y_t)
        Y_t_scaled = (Y_t - mean_y) / std_y

        # Training phase
        print("\nTraining LSTM model...")
        [lstm, dense1, dense2, mse, Y_hat_hist] = RunMyLSTM(Y_t_scaled, Y_t_scaled, 
                                        n_neurons=self.n_neurons,
                                        n_epoch=self.n_epoch, 
                                        plot_each=self.plot_each,
                                        dt=self.dt,
                                        momentum=self.momentum, 
                                        decay=self.decay,
                                        learning_rate=self.learning_rate,
                                        auto_skip=self.auto_skip)
        self.mse = float(mse)  # Convert MSE to float

        future_steps = 10
        Y_new = np.arange(len(Y_t) - 1, len(Y_t) + future_steps).reshape(-1, 1)
        print(Y_new)

        print("Shape1:", Y_hat_hist.shape)

        # Get predictions (both historical and future)
        Y_hat = ApplyMyLSTM(Y_new, lstm, dense1, dense2)
        print(Y_hat.shape)
        self.y_output = Y_hat[1:] * std_y + mean_y  # Unstandardize the predictions
        print("Shape2:", self.y_output.shape)

        # Unstandardize historical predictions properly
        Y_hat_hist = Y_hat_hist * std_y + mean_y
        Y_hat_full = np.concatenate([Y_hat_hist, self.y_output])

        # Split predictions into historical and future parts
        historical_preds = Y_hat_full[:len(Y_t)]
        future_preds = Y_hat_full[len(Y_t):]

        # Create proper timeline
        X_hist = np.arange(len(Y_t))
        X_future = np.arange(len(Y_t)-1, len(Y_t) + len(future_preds)-1)

        X_full = np.arange(0, len(X_t) + future_steps).reshape(-1, 1)
        self.X_full = X_full
        self.Y_hat_full = Y_hat_full
        self.X_hist = X_hist
        self.historical_preds = historical_preds.flatten()  # Flatten to match X_hist dimensions

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(X_hist, Y_t, 'b-', linewidth=2, label='Historical Data')
        plt.plot(X_full, Y_hat_full, 'g-', linewidth=2, label='LSTM Prediction')
        plt.axvline(x=len(Y_t)-1, color='k', linestyle='--', label='Prediction Start')
        plt.title('LSTM Model: Historical Fit and Future Predictions')
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    

        # Calculate metrics
        try:
            self.growth_percent = float(((future_preds[-1] / Y_t[-1][0]) ** (1/10) - 1) * 100)
            current_eps = float(Y_t[-1][0])
            bond_yield = 4.8
            self.intrinsic_value = float(current_eps * (7.5 + 1 * self.growth_percent) * (4.4/bond_yield))
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            self.growth_percent = 0.0
            self.intrinsic_value = 0.0

        print(f"\nGrowth Rate: {self.growth_percent:.2f}%")
        print(f"Intrinsic Value using Benjamin Graham's Formula (LSTM): ${self.intrinsic_value:.2f}")

        # Print future predictions with years
        final_historical_year = 2024
        prediction_years = [final_historical_year + i + 1 for i in range(len(future_preds))]
        print("\nPredicted values for future years:")
        for year, value in zip(prediction_years, future_preds.flatten()):
            print(f"{year}: ${value:.2f}")