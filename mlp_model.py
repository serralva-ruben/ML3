import numpy as np
from ml_utils import softmax, confusion_matrix, predict
class MLP:
    def __init__(self, input_size, hidden_size, output_size, seed=None):
        rng = np.random.default_rng(seed)
        # He initialization for weights
        self.W0 = rng.normal(hidden_size, input_size) * np.sqrt(2./input_size)
        self.b0 = np.zeros((hidden_size, 1))
        # ... (initialize other layers similarly)


    def train(self, X_train, y_train, epochs, learning_rate, batch_siz, seed=None):
        rng = np.random.default_rng(seed)

        for epoch in range(epochs):
            # Shuffle the training data
            permutation = rng.normal(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(0, X_train.shape[0], batch_size):
                # Create mini-batches
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                # Forward pass
                outputs = self.forward(X_batch)

                # Compute loss
                loss = self.compute_loss(y_batch, outputs)

                # Backpropagation to compute gradients
                gradients = self.backpropagation(X_batch, y_batch, outputs)

                # Update weights and biases based on gradients
                self.update_weights(gradients, learning_rate)

            # Optional: Print epoch number and loss for monitoring
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def forward(self, X):
        # Forward pass through the first hidden layer
        Z1 = np.dot(self.W0, X) + self.b0  # Linear step
        A1 = relu(Z1)                     # Activation step

        # Forward pass through the second hidden layer
        Z2 = np.dot(self.W1, A1) + self.b1 # Linear step
        A2 = relu(Z2)                     # Activation step

        # Forward pass through the output layer
        Z3 = np.dot(self.Wo, A2) + self.bo # Linear step
        A3 = softmax(Z3)                   # Activation step (softmax for classification)

        return A3

    def compute_loss(self, y_true, y_pred):
        # Implement the loss computation
        pass

    def backpropagation(self, X, y_true, y_pred):
        # Implement backpropagation to compute gradients
        pass

    def update_weights(self, gradients, learning_rate):
        # Update the weights and biases based on computed gradients
        pass