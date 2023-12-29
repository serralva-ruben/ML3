import numpy as np
from nn_functions import relu, relu_derivative, softmax, categorical_cross_entropy_loss
import matplotlib.pyplot as plt
from ml_utils import predict
from IPython.display import clear_output

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
        plt.show()

    def train(self, X_train, y_train, X_val, y_val, epochs, plot_interval=5):
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

            if epoch % plot_interval == 0:
                plt.clf()
                self.plot_predictions(X_val, y_val, epoch)
                plt.show()
                plt.pause(0.00001)

            

    def plot_predictions(self, X, y, current_epoch):
        # Assuming y is one-hot encoded; convert to class labels
        y_labels = np.argmax(y, axis=1) if y.ndim > 1 else y
        
        # Predict labels for X
        y_pred = predict(self, X)

        # Plotting
        self.plot_decision_boundaries(X, y)
        plt.scatter(X[:, 0], X[:, 1], c=y_labels, alpha=0.5, label='True Labels')
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.5, label='Predicted Labels')
        plt.title(f'Epoch: {current_epoch}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

    def plot_decision_boundaries(self, X, y):
        # Set min and max values and give some padding
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = 0.02  # Step size in the mesh

        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the function value for the whole grid
        Z = predict(self, np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())


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
