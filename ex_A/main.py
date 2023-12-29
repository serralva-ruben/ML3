import pandas as pd
import numpy as np
from mlp_model import MLP
from ml_utils import confusion_matrix, predict, calculate_accuracy
import matplotlib.pyplot as plt

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# Load and preprocess the datasets
def load_data(filename):
    df = pd.read_excel(filename)
    X = df[['X_0', 'X_1']].values
    y = df['y'].values
    return X, y

# Normalizing the features
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

# Load training and validation data
X_train, y_train = load_data('THA3train.xlsx')
X_val, y_val = load_data('THA3validate.xlsx')
num_classes = np.max(y_train) + 1

y_train = one_hot_encode(y_train, num_classes)
y_val = one_hot_encode(y_val, num_classes)

# Normalize features
X_train = normalize_features(X_train)
X_val = normalize_features(X_val)

# Checking values for debugging purposes
print("Sample X_train values:", X_train[:5])
print("Sample X_val values:", X_val[:5])

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

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

# Evaluate on the validation set
y_pred = predict(mlp, X_val)
print(f"Y_pred: {y_pred}")

y_pred_one_hot = one_hot_encode(y_pred, num_classes)

accuracy = calculate_accuracy(y_val, one_hot_encode(y_pred, num_classes))
print(f'Validation Accuracy: {accuracy:.2f}')

# Compute confusion matrix 
conf_matrix = confusion_matrix(y_val, y_pred_one_hot)
print('Confusion Matrix:')
print(conf_matrix)

y_val_labels = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val

plot_decision_boundaries(X_val, y_val_labels, mlp)
plt.title('Model Decision Boundaries and Validation Data')
plt.show()