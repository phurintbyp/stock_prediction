from linear_regression import EPSPrediction
import numpy as np
import matplotlib.pyplot as plt
from lstm.testCell import LSTM_test as LSTM
from RNN.RNN import RNN_test
import json

file_name = "./data/data_list/MSFT.json"

with open(file_name, 'r') as f:
    data = json.load(f)

dates = np.array(list(data.keys()))[::-1]
values = np.array(list(data.values()))[::-1]

X_t = np.arange(len(dates)).reshape(len(dates), -1)
Y_t = values.reshape(len(values), 1)

###############################################
# Linear Regression
###############################################

eps_prediction = EPSPrediction(file_name=file_name, degree=4, bond_yield=4.8)
eps_prediction.run()

###############################################
# RNN
###############################################

# rnn = RNN_test(file_name=file_name, n_epoch=700, n_neurons=100,learning_rate=1e-5, decay=0, momentum=0.95, dt=5, auto_skip=True)
# rnn.run()

###############################################
# LSTM
###############################################

lstm = LSTM(file_name=file_name, n_neurons=100, n_epoch=1000, dt=3, plot_each=100, momentum=0.98, decay=0, learning_rate=1e-4, auto_skip=True)
lstm.run()

###############################################
# Plotting and Printing out the results
###############################################

print("\nIntrinsic Value using Benjamin Graham's Formula(Linear Regression):", f"${eps_prediction.intrinsic_value:.2f}")
print("Intrinsic Value using Benjamin Graham's Formula(RNN):", f"${rnn.intrinsic_value:.2f}")
print("Intrinsic Value using Benjamin Graham's Formula(LSTM):", f"${lstm.intrinsic_value:.2f}")

print("\nGrowth Rate(Linear Regression):", f"{eps_prediction.growth_percent:.2f}%") 
print("Growth Rate(RNN):", f"{rnn.growth_percent:.2f}%")
print("Growth Rate(LSTM):", f"{lstm.growth_percent:.2f}%")

print("\nMean Squared Error(Linear Regression):", f"{float(eps_prediction.mse):.6f}")
print("Mean Squared Error(RNN):", f"{float(rnn.mse):.6f}")
print("Mean Squared Error(LSTM):", f"{float(lstm.mse):.6f}")

plt.figure(figsize=(12, 6))
plt.plot(eps_prediction.extended_years[:-10], eps_prediction.predicted_eps[:-10], linestyle="-", linewidth=2, color="red", label="Linear Regression")
# plt.plot(rnn.X_full, rnn.Y_hat_full, linestyle="-", linewidth=2, color="green", label="RNN")
plt.plot(lstm.X_full, lstm.Y_hat_full, linestyle="-", linewidth=2, color="orange", label="LSTM")
plt.plot(X_t, Y_t, linestyle="-", color="blue", linewidth=2, label='Historical Data')
plt.xlabel("Year")
plt.ylabel("EPS")
plt.title("EPS Prediction")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.show()