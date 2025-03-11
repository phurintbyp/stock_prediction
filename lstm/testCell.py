# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 03:02:45 2024

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt
import json

# Load AAPL data
with open('./data/data_list/USB.json', 'r') as f:
    data = json.load(f)

# Convert to numpy array
Y_t = np.array(list(data.values()))[::-1]  # Reverse to get chronological order
Y_t = Y_t.reshape(len(Y_t), 1)

# Plot the original data
plt.plot(Y_t)
plt.title('Earnings Per Share')
plt.show()

###############################################################################
#forecast Y(t) --> Y(t + dt)
from LSTM import *

dt = 10  # Predict next year
[lstm, dense1, dense2] = RunMyLSTM(Y_t, Y_t, n_neurons=50,
                                   n_epoch=1000, plot_each=50, dt=dt,
                                   momentum=0.95, decay=0.001,
                                   learning_rate=5e-4,
                                   auto_skip=True)  # Add auto_skip parameter

Y_hat = ApplyMyLSTM(Y_t, lstm, dense1, dense2)
    
X_plot = np.arange(0, len(Y_t))
X_plot_hat = np.arange(0, len(Y_hat)) + dt

plt.plot(X_plot, Y_t)
plt.plot(X_plot_hat, Y_hat)
plt.legend(['Actual EPS', 'Predicted EPS'])
plt.title('Earnings Prediction')
plt.show()
###############################################################################

growth_percent = ((Y_hat[-1] / Y_t[-1]) ** (1/dt) - 1) * 100
current_eps = Y_t[-1][0]
bond_yield = 4.8

print(current_eps)
print(growth_percent)

intrinsic_value = current_eps * (7.5 + 1 * growth_percent) * (4.4/bond_yield)
print("Intrinsic Value using Benjamin Graham's Formula(LSTM):", intrinsic_value)