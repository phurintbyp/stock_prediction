import numpy as np
import matplotlib.pyplot as plt
import json
from MyRNN import *

np.random.seed(31)

def close_on_key(event):
    """Close the figure if the '0' key is pressed."""
    if event.key == '0':
        plt.close(event.canvas.figure)

# 1) Load data
with open('./data_list/AMD.json', 'r') as f:
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
plt.title("AAPL Historical Data (Unscaled)")
plt.xlabel("Time Index")
plt.ylabel("Value")
plt.legend()
plt.show()

# 6) Train RNN on the SCALED Y data
rnn = RunMyRNN(
    X_t,
    Y_t_scaled,          # <--- use scaled target here
    Activation=Tanh(),
    n_epoch=500,         # can increase more if needed
    n_neurons=400,        # fewer neurons for small dataset
    learning_rate=2e-5,  # higher LR to avoid slow/stuck training
    decay=0,          # tiny weight decay
    momentum=0.9,        
    dt=0
)

# 7) Predict next 10 steps using the trained model
future_steps = 10
X_new = np.arange(len(X_t), len(X_t) + future_steps).reshape(-1, 1)

# 8) Get scaled predictions
Y_new_scaled = ApplyMyRNN(X_new, rnn)

# 9) Invert the scaling for final predictions: Y_new = Y_scaled * std_y + mean_y
Y_new = (Y_new_scaled * std_y) + mean_y

# 10) Plot historical vs. predicted
plt.figure(figsize=(10, 6))
plt.plot(X_t, Y_t, 'b-', label='Historical Data')
plt.plot(X_new, Y_new, 'r--', label='10-Year Prediction')
plt.title("AAPL Value Prediction (Scaled RNN)")
plt.xlabel("Time Index (Years since 1996)")
plt.ylabel("Value")
plt.legend()

# Add vertical line to show where prediction starts
pred_start = len(X_t) - 1
plt.axvline(x=pred_start, color='k', linestyle=':', alpha=0.5)
plt.text(pred_start + 0.5, np.min(Y_t),
         'Prediction Start', rotation=90)

fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', close_on_key)
plt.show()

# 11) Print out the future predictions
final_historical_year = 2024
prediction_years = [final_historical_year + i + 1 for i in range(future_steps)]
print("Predicted values for future years:")
for year, value in zip(prediction_years, Y_new.flatten()):
    print(f"{year}: {value:.2f}")