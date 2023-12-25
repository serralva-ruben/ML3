import pandas as pd
import numpy as np
from mlp_model import MLP
from ml_utils import confusion_matrix, predict, calculate_accuracy

def one_hot_encode(y):
    return np.array([[1, 0] if label == 0 else [0, 1] for label in y])

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

y_train = one_hot_encode(y_train)
y_val = one_hot_encode(y_val)

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
learning_rate = 0.01
batch_size = 32
epochs = 50

mlp = MLP(input_size, hidden_size, output_size, learning_rate, batch_size)

# Train the model
mlp.train(X_train, y_train, epochs)

# Evaluate on the validation set
y_pred = predict(mlp, X_val)
accuracy = calculate_accuracy(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

# Compute confusion matrix 
conf_matrix = confusion_matrix(y_val, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

