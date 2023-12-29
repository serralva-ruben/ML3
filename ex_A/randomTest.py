import numpy as np
from mlp_model import MLP
from sklearn.model_selection import train_test_split
from ml_utils import confusion_matrix, calculate_accuracy
import matplotlib.pyplot as plt

# Function to generate random dataset
def generate_random_data(num_samples, num_features):
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, 2, size=(num_samples, ))
    return X, y

# Function to normalize features
def normalize_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def plot_decision_boundaries(X, y, model):
    # Set min and max values and give some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02  # Step size in the mesh

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = predict(model, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Generate random data
num_samples = 10000
num_features = 2
X, y = generate_random_data(num_samples, num_features)

# Normalize features
X = normalize_features(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the labels if your model expects one-hot encoded labels
y_train = np.eye(2)[y_train]
y_test = np.eye(2)[y_test]

# Initialize the MLP model
input_size = 2
hidden_size = 10
output_size = 2
learning_rate = 0.00001
batch_size = 2
epochs = 100

mlp = MLP(input_size, hidden_size, output_size, learning_rate, batch_size)

# Train the model
mlp.train(X_train, y_train, epochs)

# Function to predict labels
def predict(model, X):
    outputs = model.forward(X)
    y_pred = np.argmax(outputs, axis=0)
    return y_pred

# Evaluate on the test set
y_pred = predict(mlp, X_test)

y_test_labels = np.argmax(y_test, axis=1)

# Compute accuracy and confusion matrix
accuracy = calculate_accuracy(y_test_labels, y_pred)
confusion_matrix = confusion_matrix(y_test_labels, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(confusion_matrix)

y_val_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

plot_decision_boundaries(y_test, y_val_labels, mlp)
plt.title('Model Decision Boundaries and Validation Data')
plt.show()