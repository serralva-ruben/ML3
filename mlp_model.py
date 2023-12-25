import numpy as np
from nn_functions import relu, relu_derivative, softmax, binary_cross_entropy_loss
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
        rng = np.random.default_rng()
        for epoch in range(epochs):
            # Shuffle the training data
            permutation = rng.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(0, X_train.shape[0], self.batch_size):
                # Create mini-batches
                X_batch = X_train_shuffled[i:i + self.batch_size]
                y_batch = y_train_shuffled[i:i + self.batch_size]

                # Forward pass (transposing X_batch beforehand)
                X_batch = X_batch.T
                outputs = self.forward(X_batch)

                # Debugging: Print the min and max of the outputs
                print("Output range in this batch: min =", np.min(outputs), ", max =", np.max(outputs))

                # Compute loss
                loss = binary_cross_entropy_loss(y_batch, outputs)

                # Backpropagation to compute gradients
                gradients = self.backpropagation(X_batch, y_batch, outputs)

                # Update weights and biases based on gradients
                self.update_weights(gradients)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")


    def forward(self, X):
        # Forward pass through the first hidden layer
        Z1 = np.dot(self.W0, X) + self.b0  # Linear step
        self.A1 = relu(Z1)                 # Activation step

        # Forward pass through the second hidden layer
        Z2 = np.dot(self.W1, self.A1) + self.b1  # Linear step
        self.A2 = relu(Z2)                        # Activation step

        # Forward pass through the output layer
        Z3 = np.dot(self.Wo, self.A2) + self.bo   # Linear step
        A3 = softmax(Z3)                          # Activation step (softmax for classification)

        return A3

    def backpropagation(self, X, y_true, y_pred):
        # Gradients dictionary
        gradients = {}

        # Error at output layer
        dZo = y_pred - y_true  # For binary cross-entropy loss and softmax activation
        dWo = np.dot(dZo, self.A2.T)
        dbo = np.sum(dZo, axis=1, keepdims=True)

        # Error at second hidden layer
        dA2 = np.dot(self.Wo.T, dZo)
        dZ2 = dA2 * relu_derivative(self.A2)
        dW1 = np.dot(dZ2, self.A1.T)
        db1 = np.sum(dZ2, axis=1, keepdims=True)

        # Error at first hidden layer
        dA1 = np.dot(self.W1.T, dZ2)
        dZ1 = dA1 * relu_derivative(self.A1)
        dW0 = np.dot(dZ1, X.T)
        db0 = np.sum(dZ1, axis=1, keepdims=True)

        gradients['Wo'] = dWo
        gradients['bo'] = dbo
        gradients['W1'] = dW1
        gradients['b1'] = db1
        gradients['W0'] = dW0
        gradients['b0'] = db0

        return gradients


    def update_weights(self, gradients):
        self.Wo -= self.learning_rate * gradients['Wo']
        self.bo -= self.learning_rate * gradients['bo']
        self.W1 -= self.learning_rate * gradients['W1']
        self.b1 -= self.learning_rate * gradients['b1']
        self.W0 -= self.learning_rate * gradients['W0']
        self.b0 -= self.learning_rate * gradients['b0']
