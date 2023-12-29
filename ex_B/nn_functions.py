# for the hidden layers φ(0) and φ(0) we used ReLU
# for the output layer φ(2) used Sigmoid
import numpy as np
#relu function for hidden layers
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

#sigmoid function output layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def categorical_cross_entropy_loss(y_true, y_pred):
    m = y_pred.shape[0]  # Number of samples
    loss = -1/m * np.sum(y_true * np.log(y_pred + 1e-10))
    return loss

