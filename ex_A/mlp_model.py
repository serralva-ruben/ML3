import numpy as np
from nn_functions import relu, relu_derivative, softmax, categorical_cross_entropy_loss
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, batch_size=32, seed=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        rng = np.random.default_rng(seed)
        
        # He initialization for weights in layers with ReLU activation
        # First hidden layer
        self.W0 = rng.normal(loc=0, scale=np.sqrt(2./input_size), size=(hidden_size, input_size))
        self.b0 = np.zeros((hidden_size, 1))
        
        # Second hidden layer
        self.W1 = rng.normal(loc=0, scale=np.sqrt(2./hidden_size), size=(hidden_size, hidden_size))
        self.b1 = np.zeros((hidden_size, 1))

        # Output layer (assuming Softmax activation)
        self.Wo = rng.normal(loc=0, scale=np.sqrt(2./hidden_size), size=(output_size, hidden_size))
        self.bo = np.zeros((output_size, 1))

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
        # Shuffle the training data if needed
        # (optional, but often beneficial for stochastic gradient descent)

            for i in range(0, len(y_train), self.batch_size):
                # Mini-batch training
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss (implement a function to calculate the loss)
                loss = categorical_cross_entropy_loss(y_batch, y_pred.T)

                # Backpropagation to compute gradients
                gradients = self.backpropagation(X_batch, y_batch, y_pred)

                # Update weights and biases
                self.update_weights(gradients)

            if epoch % 2 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")


    def forward(self, X):
        Z1 = np.dot(self.W0, X.T) + self.b0
        self.A1 = relu(Z1)

        Z2 = np.dot(self.W1, self.A1) + self.b1
        self.A2 = relu(Z2)

        Z3 = np.dot(self.Wo, self.A2) + self.bo
        A3 = softmax(Z3)

        return A3

    def backpropagation(self, X, y_true, y_pred):
        # Number of samples
        m = y_true.shape[1]

        # Error at output layer (Softmax and cross-entropy)
        dZo = y_pred - y_true.T
        dWo = (1/m) * np.dot(dZo, self.A2.T)
        dbo = (1/m) * np.sum(dZo, axis=1, keepdims=True)

        # Error at second hidden layer (ReLU activation)
        dA2 = np.dot(self.Wo.T, dZo)
        dZ2 = dA2 * relu_derivative(self.A2)  # Element-wise multiplication
        dW1 = (1/m) * np.dot(dZ2, self.A1.T)
        db1 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Error at first hidden layer (ReLU activation)
        dA1 = np.dot(self.W1.T, dZ2)
        dZ1 = dA1 * relu_derivative(self.A1)  # Element-wise multiplication
        dW0 = (1/m) * np.dot(dZ1, X)
        db0 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Store gradients
        gradients = {'Wo': dWo, 'bo': dbo, 'W1': dW1, 'b1': db1, 'W0': dW0, 'b0': db0}

        return gradients

    def update_weights(self, gradients):
        self.Wo -= self.learning_rate * gradients['Wo']
        self.bo -= self.learning_rate * gradients['bo']
        self.W1 -= self.learning_rate * gradients['W1']
        self.b1 -= self.learning_rate * gradients['b1']
        self.W0 -= self.learning_rate * gradients['W0']
        self.b0 -= self.learning_rate * gradients['b0']
