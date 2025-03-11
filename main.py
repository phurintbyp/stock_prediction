from linear_regression import EPSPrediction
import numpy as np
import matplotlib.pyplot as plt
from lstm.testCell import LSTM_test as LSTM
from RNN.RNN import RNN

###############################################
# Linear Regression
###############################################

file_name = "./data/data_list/USB.json"
degree = 2
bond_yield = 4.8
eps_prediction = EPSPrediction(file_name, degree, bond_yield)
eps_prediction.run()
print("Mean Squared Error:", eps_prediction.MSE())
print("R-Squared:", eps_prediction.RSQ())

###############################################
# RNN
###############################################

rnn = RNN(file_name="./data/data_list/USB.json", n_epoch=500, n_neurons=400, learning_rate=2e-5, decay=0, momentum=0.9, dt=0)
rnn.run()

###############################################
# LSTM
###############################################

lstm = LSTM(file_name="./data/data_list/USB.json", n_neurons=50, n_epoch=1000, plot_each=100, dt=10, momentum=0.95, decay=0.001, learning_rate=5e-4, auto_skip=True)
lstm.run()