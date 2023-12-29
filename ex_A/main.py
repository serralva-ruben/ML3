import pandas as pd
import numpy as np
from mlp_model import MLP
from ml_utils import confusion_matrix, predict, calculate_accuracy
import matplotlib.pyplot as plt

plt.ion()

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

# Load training and validation data
X_train, y_train = load_data('THA3train.xlsx')
X_val, y_val = load_data('THA3validate.xlsx')
num_classes = np.max(y_train) + 1

y_train = one_hot_encode(y_train, num_classes)
y_val = one_hot_encode(y_val, num_classes)

# Normalize features
X_train = normalize_features(X_train)
X_val = normalize_features(X_val)

# Initialize the MLP model
input_size = 2
hidden_size = 10
output_size = 2
learning_rate = 0.000001
batch_size = 16
epochs = 1000

mlp = MLP(input_size, hidden_size, output_size, learning_rate, batch_size)

# Train the model
mlp.train(X_train, y_train, X_val, y_val, epochs, plot_interval=1)

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