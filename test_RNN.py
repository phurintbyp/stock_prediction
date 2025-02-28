import numpy as np
import matplotlib.pyplot as plt
from RNN_cell import *

# Creates random sine shaped data
X_t = np.arange(-10,10,0.1)
X_t = X_t.reshape(len(X_t),1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t),1)

# Plots data
plt.plot(X_t, Y_t)
plt.show()

n_neurons = 500

rnn   = RNN(X_t, n_neurons, Tanh())
optimizer = Optimizer_SGD(learning_rate=1e-5, decay=0.01, momentum = 0.9)
T = rnn.T
n_epochs = 200

Monitor = np.zeros((n_epochs, 1))

for n in range(n_epochs):

    rnn.forward()

    Y_hat = rnn.Y_hat
    dY = Y_hat - Y_t
    L = 0.5 * np.dot(dY.T, dY)/T

    print(L)
    Monitor[n] = L

    rnn.backward(dY)

    optimizer.pre_update_params()
    optimizer.update_params(rnn)
    optimizer.post_update_params()

    # Plots data and prediction
    if not n % 100:
        plt.plot(X_t, Y_t, color='blue') # Plots actual data
        plt.plot(X_t, Y_hat, color='orange') # Plots predicted data
        plt.title('epoch ' + str(n))
        plt.legend(['y', '$\hat{y}$'])
        plt.show()
    

# Plots data and prediction
plt.plot(X_t, Y_t, color='blue') # Plots actual data
plt.plot(X_t, Y_hat, color='orange') # Plots predicted data
plt.title('epoch ' + str(n))
plt.legend(['y', '$\hat{y}$'])
plt.show()

plt.plot(range(n_epochs), Monitor)
plt.xlabel('epochs')
plt.ylabel('MSSE')
plt.yscale('log')
plt.show()