# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def close_on_key(event):
    """Close the current figure if the '0' key is pressed."""
    if event.key == '0':
        plt.close(event.canvas.figure)

def RunMyRNN(X_t, Y_t, Activation, n_epoch=500, n_neurons=400,
             learning_rate=1e-5, decay=0.0, momentum=0.8,
             plot_each=100, dt=0, auto_skip=False):
    
    # initializing RNN
    rnn = RNN(n_neurons, Activation)
    optimizer = Optimizer_SGD(learning_rate, decay, momentum)
    T = max(X_t.shape)
    X_plot = np.arange(0, T)
    
    # Separate the training subset (X_t_dt, Y_t_dt) vs. the full series (X_t, Y_t)
    if dt != 0:
        X_t_dt = Y_t[:-dt]   # for input
        Y_t_dt = Y_t[dt:]    # for target
        X_plots = X_plot[dt:]
    else:
        X_t_dt = X_t
        Y_t_dt = Y_t
        X_plots = X_plot
    
    print("RNN is running...")
    
    for n in range(n_epoch):
        # 1) Forward on training subset
        rnn.forward(X_t_dt)
        # 2) Compute training-subset error
        dY = rnn.Y_hat - Y_t_dt
        # 3) Backprop & update
        rnn.backward(dY)
        optimizer.pre_update_params()
        optimizer.update_params(rnn)
        optimizer.post_update_params()
        
        # 4) Every 'plot_each' epochs, measure & print both partial and full-series errors
        if not n % plot_each:
            # --- Partial (training-subset) error ---
            L = 0.5 * np.dot(dY.T, dY) / (T - dt)
            
            # --- Full-series error ---
            # Forward pass on the entire X_t
            rnn.forward(X_t)
            Y_hat_epoch = ApplyMyRNN(Y_t, rnn)  # Same as final graph
            
            # Plot the full-series predictions
            M = np.max(np.vstack((rnn.Y_hat, Y_t)))
            m = np.min(np.vstack((rnn.Y_hat, Y_t)))
            
            plt.plot(X_plot, Y_t)
            plt.plot(X_plot + dt, Y_hat_epoch)  # Predictions starting from dt onwards
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(['y', '$\\hat{y}$'])
            plt.title(f'epoch {n}')
            
            if dt != 0:
                plt.fill_between([X_plot[-1], X_plots[-1] + dt], m, M, color='k', alpha=0.1)
                plt.plot([X_plot[-1], X_plot[-1]], [m, M], 'k-', linewidth=3)
            
            fig = plt.gcf()
            fig.canvas.mpl_connect('key_press_event', close_on_key)
            plt.show(block=not auto_skip)
            
            if auto_skip:
                plt.pause(1)
                plt.close()
            
            print(f'epoch={n}, MSSE={L[0,0]:.3f}')
    
    # ========== Final forward pass on full data ==========
    rnn.forward(X_t)
    if dt != 0:
        dY_full = rnn.Y_hat[:-dt] - Y_t[dt:]
        L_final = 0.5 * np.dot(dY_full.T, dY_full) / (T - dt)
    else:
        dY_full = rnn.Y_hat - Y_t
        L_final = 0.5 * np.dot(dY_full.T, dY_full) / T
    
    # Plot final result
    M = np.max(np.vstack((rnn.Y_hat, Y_t)))
    m = np.min(np.vstack((rnn.Y_hat, Y_t)))
    
    plt.plot(X_plot, Y_t)
    plt.plot(X_plot + dt, Y_hat_epoch)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['y', '$\\hat{y}$'])
    plt.title(f'epoch {n}')
    
    if dt != 0:
        plt.fill_between([X_plot[-1], X_plots[-1] + dt], m, M, color='k', alpha=0.1)
        plt.plot([X_plot[-1], X_plot[-1]], [m, M], 'k-', linewidth=3)
    
    fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', close_on_key)
    plt.show()
    
    print(f'Done! MSSE (full-series) = {L_final[0,0]:.3f}')
    
    return (rnn, L[0,0])

def ApplyMyRNN(X_t, rnn):
    T = max(X_t.shape)
    Y_hat = np.zeros((T, 1))
    
    # Re-initialize hidden states
    H = [np.zeros((rnn.n_neurons, 1)) for _ in range(T + 1)]
    ht = H[0]
    
    _, _, Y_hat = rnn.RNNCell(X_t, ht, rnn.ACT, H, Y_hat)
    return Y_hat

class RNN:
    def __init__(self, n_neurons, Activation):
        self.n_neurons = n_neurons
        
        self.Wx = 0.1 * np.random.randn(n_neurons, 1)
        self.Wh = 0.1 * np.random.randn(n_neurons, n_neurons)
        self.Wy = 0.1 * np.random.randn(1, n_neurons)
        self.biases = 0.1 * np.random.randn(n_neurons, 1)
        
        self.Activation = Activation
    
    def forward(self, X_t):
        self.T = max(X_t.shape)
        self.X_t = X_t
        
        # Prediction vector of y
        self.Y_hat = np.zeros((self.T, 1))
        # Hidden states
        self.H = [np.zeros((self.n_neurons, 1)) for _ in range(self.T + 1)]
        
        # Initialize gradients to zero
        self.dWx = np.zeros((self.n_neurons, 1))
        self.dWh = np.zeros((self.n_neurons, self.n_neurons))
        self.dWy = np.zeros((1, self.n_neurons))
        self.dbiases = np.zeros((self.n_neurons, 1))
        
        # RNN cell pass
        ACT = [self.Activation for _ in range(self.T)]
        ht = self.H[0]  # initial state
        ACT, H, Y_hat = self.RNNCell(X_t, ht, ACT, self.H, self.Y_hat)
        
        self.Y_hat = Y_hat
        self.H = H
        self.ACT = ACT
    
    def RNNCell(self, X_t, ht, ACT, H, Y_hat):
        for t, xt in enumerate(X_t):
            xt = xt.reshape(1, 1)
            out = np.dot(self.Wx, xt) + np.dot(self.Wh, ht) + self.biases
            ACT[t].forward(out)
            ht = ACT[t].output
            y_hat_t = np.dot(self.Wy, ht)
            H[t+1] = ht
            Y_hat[t] = y_hat_t
        return ACT, H, Y_hat
    
    def backward(self, dvalues):
        T = self.T
        H = self.H
        X_t = self.X_t
        ACT = self.ACT
        
        dWx = self.dWx
        dWy = self.dWy
        dWh = self.dWh
        dbiases = self.dbiases
        
        Wy = self.Wy
        Wh = self.Wh
        
        # Start with the final time step
        dht = np.dot(Wy.T, dvalues[-1].reshape(1, 1))
        
        # BPTT
        for t in reversed(range(T)):
            dy = dvalues[t].reshape(1, 1)
            xt = X_t[t].reshape(1, 1)
            
            ACT[t].backward(dht)
            dtanh = ACT[t].dinputs
            
            # Gradients
            dWx += np.dot(dtanh, xt)
            dWy += np.dot(H[t+1], dy).T
            dWh += np.dot(H[t], dtanh.T)
            dbiases += dtanh
            
            dht = np.dot(Wh, dtanh) + np.dot(Wy.T, dy)
        
        self.dWx = dWx
        self.dWy = dWy
        self.dWh = dWh
        self.dbiases = dbiases

class Tanh:
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        self.inputs = inputs
    
    def backward(self, dvalues):
        deriv = 1 - self.output**2
        self.dinputs = deriv * dvalues

class Optimizer_SGD:
    def __init__(self, learning_rate=1e-5, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = (
                self.learning_rate *
                (1 / (1 + self.decay * self.iterations))
            )
    
    def update_params(self, layer):
        if self.momentum:
            # Initialize momentum arrays if not already
            if not hasattr(layer, 'Wx_momentums'):
                layer.Wx_momentums = np.zeros_like(layer.Wx)
                layer.Wy_momentums = np.zeros_like(layer.Wy)
                layer.Wh_momentums = np.zeros_like(layer.Wh)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            Wx_updates = self.momentum * layer.Wx_momentums \
                         - self.current_learning_rate * layer.dWx
            layer.Wx_momentums = Wx_updates
            
            Wy_updates = self.momentum * layer.Wy_momentums \
                         - self.current_learning_rate * layer.dWy
            layer.Wy_momentums = Wy_updates
            
            Wh_updates = self.momentum * layer.Wh_momentums \
                         - self.current_learning_rate * layer.dWh
            layer.Wh_momentums = Wh_updates
            
            bias_updates = self.momentum * layer.bias_momentums \
                           - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            Wx_updates = -self.current_learning_rate * layer.dWx
            Wy_updates = -self.current_learning_rate * layer.dWy
            Wh_updates = -self.current_learning_rate * layer.dWh
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        # Update parameters
        layer.Wx += Wx_updates
        layer.Wy += Wy_updates
        layer.Wh += Wh_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.iterations += 1