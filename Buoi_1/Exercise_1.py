import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# data = pd.read_csv('Advertising.csv')
# X = data.values[:, 2]
# Y = data.values[:, 4]

data = pd.read_csv('Data.csv')
X = data.values[:, 1]
Y = data.values[:, 0]

# plt.scatter(X, Y, marker = 'o')
# plt.show()

def predict(X, weight, bias):
    return X*weight + bias

def cost_count(X, Y, weight, bias):
    num_of_mem = len(X)
    total_error = 0.0
    for i in range(num_of_mem):
        total_error += (Y[i] - (weight*X[i] + bias))**2
    return total_error/(2 * num_of_mem)

def update_param(X, Y, weight, bias, learning_rate):
    num_of_mem = len(X)
    temp_weight = 0.0
    temp_bias = 0.0
    for i in range(num_of_mem):
        temp_weight += -2 * (Y[i] - (weight * X[i] + bias)) * X[i]
        temp_bias   += -2 * (Y[i] - (weight * X[i] + bias))
    weight -= (temp_weight / (2 * num_of_mem)) * learning_rate
    bias -= (temp_bias / (2 * num_of_mem)) * learning_rate
    return weight, bias

def training(X, Y, weight, bias, learning_rate, timeout):
    cost_list = []
    for i in range(timeout):
        weight, bias = update_param(X, Y, weight, bias, learning_rate)
        cost = cost_count(X, Y, weight, bias)
        cost_list.append(cost)
    return weight, bias, cost_list

# Training model
# 1. Using gradient desent
weight, bias, cost = training(X, Y, 0.1, 98.24, 0.00000001, 10000)
print('\nResult using gradient desent: \n + weight: {0}\n + bias: {1}\n'.format(weight, bias))

t = [i for i in range(len(cost))]
plt.subplot(121)
plt.plot(t, cost)

X0 = np.linspace(1000, 2500, 2)
Y0 = bias + weight*X0
plt.subplot(122)
plt.plot(X, Y, 'ro')
plt.plot(X0, Y0)

reg = LinearRegression().fit(X.T.reshape(-1, 1), Y.T.reshape(-1, 1))
plt.plot(X0, reg.coef_[0]*X0 + reg.intercept_[0], 'y--')
print('\nResult using scikit learn lib: \n + weight: {0}\n + bias: {1}\n'.format(reg.coef_[0][0], reg.intercept_[0]))

plt.axis([900, 2600, 50, 500])
plt.show()