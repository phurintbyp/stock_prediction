import numpy as np
import matplotlib.pyplot as plt
import json
from MyRNN import *

def close_on_key(event):
    """Close the figure if the '0' key is pressed."""
    if event.key == '0':
        plt.close(event.canvas.figure)

with open('./data_list/AAPL.json', 'r') as f:
    data = json.load(f)

dates = np.array(list(data.keys()))[::-1]  # Reverse order of dates
values = np.array(list(data.values()))[::-1]  # Reverse order of values

X_t = np.arange(len(dates)).reshape(len(dates), -1)
Y_t = values.reshape(len(values), 1)

# 1. Plot the original data
plt.plot(X_t, Y_t)
plt.title("Original Data")
plt.xlabel("Time")
plt.ylabel("Value")

fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', close_on_key)  # Connect key-press callback
plt.show()

# 2. Train the RNN on the original data
rnn = RunMyRNN(X_t, Y_t, Tanh())

# 3. Predict 10 new points
X_new = np.arange(len(dates), len(dates) + 10).reshape(10, 1)
Y_hat = ApplyMyRNN(X_new, rnn)

# 4. Plot the predictions
plt.plot(X_t, Y_t, label='Original')
plt.plot(X_new, Y_hat, label='Predicted')
plt.legend()
plt.title("RNN Predictions")
plt.xlabel("Time")
plt.ylabel("Value")

fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', close_on_key)
plt.show()

# # 5. Retrain with a time shift
# dt = 1
# rnn = RunMyRNN(Y_t, Y_t, Tanh(), n_epoch=800, n_neurons=10, decay=0.1, dt=dt)
# Y_hat = ApplyMyRNN(Y_t, rnn)

# X_t = np.arange(len(Y_t))

# # 6. Plot retrained predictions
# plt.plot(X_t, Y_t, label='Original')
# plt.plot(X_t + dt, Y_hat, label='Predicted')
# plt.legend()
# plt.title("RNN Retrained Predictions")
# plt.xlabel("Time")
# plt.ylabel("Value")

fig = plt.gcf()
fig.canvas.mpl_connect('key_press_event', close_on_key)
plt.show()