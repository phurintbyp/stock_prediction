import numpy as np
import matplotlib.pyplot as plt

class Linear_regression:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, x, y, degree):
        x_reshaped = x.reshape(-1, 1)
        y_reshaped = y.reshape(-1, 1)
        
        ones = np.ones((x.shape[0], 1))

        power = np.array([]).reshape(x.shape[0], 0)

        if (degree > 1):
            for i in range(2, degree + 1) :
                power = np.concatenate((x_reshaped**i, power), axis = 1)

        concate_x = np.concatenate((power, x_reshaped, ones), axis=1)

        x_transpose = concate_x.T
        a = (np.linalg.inv(x_transpose @ concate_x)) @ (x_transpose @ y_reshaped)
        
        self.coef = a[0:-1]
        self.intercept = a[-1]

    def predict(self, x, degree):
        y = 0
        for i in range(0, degree):
            y += self.coef[i]*(x**(degree-i))

        return y + self.intercept

    def create_graph(self, x, y, degree=1):
        self.fit(x, y, degree)
        plt.scatter(x, y, label="Data")
        plt.plot(x, self.predict(x, degree), 'r-', label="Regression Line")
        np.set_printoptions(suppress=True, precision=5)
        print(self.coef)
        print(self.intercept)
        
        plt.legend()
        plt.show()
