from linear_regression import EPSPrediction
import numpy as np
import matplotlib.pyplot as plt
from lstm.testCell import LSTM_test as LSTM
from RNN.RNN import RNN_test

file_name = "./data/data_list/AAPL.json"

###############################################
# Linear Regression
###############################################

eps_prediction = EPSPrediction(file_name=file_name, degree=2, bond_yield=4.8)
eps_prediction.run()

###############################################
# RNN
###############################################

rnn = RNN_test(file_name=file_name, n_epoch=500, n_neurons=400, learning_rate=2e-5, decay=0, momentum=0.9, dt=0, auto_skip=True)
rnn.run()

###############################################
# LSTM
###############################################

lstm = LSTM(file_name=file_name, n_neurons=50, n_epoch=500, dt=10, plot_each=100, momentum=0.95, decay=0.001, learning_rate=5e-4, auto_skip=True)
lstm.run()

###############################################
# Printing out the results
###############################################

print("\nIntrinsic Value using Benjamin Graham's Formula(Linear Regression):", f"{eps_prediction.intrinsic_value:.2f}")
print("Intrinsic Value using Benjamin Graham's Formula(RNN):", f"{rnn.intrinsic_value:.2f}")
print("Intrinsic Value using Benjamin Graham's Formula(LSTM):", f"{lstm.intrinsic_value[0]:.2f}")

print("\nMean Squared Error(Linear Regression):", eps_prediction.mse[0])
print("Mean Squared Error(RNN):", rnn.mse)
print("Mean Squared Error(LSTM):", lstm.mse)