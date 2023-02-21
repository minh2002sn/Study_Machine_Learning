import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_classification.csv', header=None).values
N, d = data.shape

true_x = []
true_y = []
false_x = []
false_y = []

for item in data:
    if item[2] == 1:
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_x.append(item[0])
        false_y.append(item[1])

def sigmoid(z):
    return 1 / (1 + np.exp(z))

def classify(value):
    if p >= 0.5:
        return 1
    else:
        return 0
    
def predict(features, weights):
    return sigmoid(np.dot(features, weights))

def cost_function(features, weights, labels):
    z = predict(features, weights)
    cost = -labels*np.log(z) - (1 - labels)*np.log(1 - z)
    return np.sum(cost)

def update_weights(features, weights, labels, learning_rate):
    z = predict(features, weights)
    weights -= learning_rate*np.dot(features.T, (z - labels))/N
    return weights

def training(features, weights, labels, learning_rate, timeout):
    for i in range(timeout):
        weights = update_weights(features, weights, labels, learning_rate)
        print("{}. Lost = {}".format(i, cost_function(features, weights, labels)))
    return weights

x = data[:, 0:d - 1].reshape(-1, d - 1)
labels = data[:, -1].reshape(-1, 1)
features = np.hstack((np.ones((N, 1)), x))

weights = np.array([0., 0.1, 0.1]).reshape(-1, 1)
weights = training(features, weights, labels, 0.01, 100000)

print(weights)

plt.scatter(true_x, true_y, marker='o', c='b')
plt.scatter(false_x, false_y, marker='s', c='r')
plt.show()
