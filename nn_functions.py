# for the hidden layer φ(0) and φ(0) we can use ReLU
# for the output layer φ(2) we can use Softmax
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def binary_cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]  # Number of samples
    loss = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
 