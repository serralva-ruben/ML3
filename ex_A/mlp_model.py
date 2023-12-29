import numpy as np
from nn_functions import relu, relu_derivative, categorical_cross_entropy_loss, sigmoid
import matplotlib.pyplot as plt
from ml_utils import predict

class MLP:
    def __init__(self, input_size=2, hidden_size=10, output_size=2, learning_rate=0.01, batch_size=32, seed=42):
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

    def train(self, X_train, y_train, X_val, y_val, epochs, plot_interval=5, early_stopping_threshold=0.0001):
        train_losses = []
        val_losses = []

        plt.figure(figsize=(8, 6))
        live_plot_figure = plt.gcf() 

        for epoch in range(epochs):
            # Training phase
            train_loss = self.run_epoch(X_train, y_train, training=True)
            train_losses.append(train_loss)

            # Validation phase
            val_loss = self.run_epoch(X_val, y_val, training=False)
            val_losses.append(val_loss)

            # Early stopping based on validation loss
            if epoch > 0 and abs(val_losses[-1] - val_losses[-2]) < early_stopping_threshold:
                print(f"Stopping early at epoch {epoch+1}")
                break

            # Plotting
            if epoch % plot_interval == 0:
                live_plot_figure.clf()
                self.plot_decision_boundaries(X_val, y_val, epoch)
                live_plot_figure.canvas.draw()
                plt.pause(0.01)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        return train_losses, val_losses

    def run_epoch(self, X, y, training=True):
        epoch_loss = 0
        num_batches = int(np.ceil(len(y) / self.batch_size))

        for i in range(0, len(y), self.batch_size):
            X_batch = X[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]

            # Forward pass
            y_pred = self.forward(X_batch)

            # Compute loss
            loss = categorical_cross_entropy_loss(y_batch, y_pred.T)
            epoch_loss += loss

            if training:
                # Backpropagation and weight update
                gradients = self.backpropagation(X_batch, y_batch, y_pred)
                self.update_weights(gradients)

        return epoch_loss / num_batches

    def plot_decision_boundaries(self, X, y, current_epoch):
        # Assuming y is one-hot encoded; convert to class labels
        y_labels = np.argmax(y, axis=1) if y.ndim > 1 else y

        # Plotting decision boundaries
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # Predict the function value for the whole grid
        Z = predict(self, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        # Separate each class
        class_0 = X[y_labels == 0]
        class_1 = X[y_labels == 1]

        #update the plot title and scatter the true points for each class and legend
        plt.title(f'Epoch: {current_epoch+1}')
        plt.scatter(class_0[:, 0], class_0[:, 1], alpha=1, edgecolor='k', label='Class 0', cmap='winter', color= 'Purple')
        plt.scatter(class_1[:, 0], class_1[:, 1], alpha=1, edgecolor='k', label='Class 1', cmap='winter', color = 'Yellow')
        plt.legend()

    def forward(self, X):
        Z1 = np.dot(self.W0, X.T) + self.b0
        self.A1 = relu(Z1)

        Z2 = np.dot(self.W1, self.A1) + self.b1
        self.A2 = relu(Z2)

        Z3 = np.dot(self.Wo, self.A2) + self.bo
        A3 = sigmoid(Z3)

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
