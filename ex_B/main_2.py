import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlp_model_2 import MLP
from ml_utils import confusion_matrix, predict, calculate_accuracy

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def load_data(filename):
    df = pd.read_excel(filename)
    X = df[['X_0', 'X_1']].values
    y = df['y'].values
    return X, y

def normalize_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Load and preprocess data
X_train, y_train = load_data('../THA3train.xlsx')
X_val, y_val = load_data('../THA3validate.xlsx')
num_classes = np.max(y_train) + 1

y_train = one_hot_encode(y_train, num_classes)
y_val = one_hot_encode(y_val, num_classes)
X_train = normalize_features(X_train)
X_val = normalize_features(X_val)

# Hyperparameters for experiments
learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
std_devs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Setup for heatmap
heatmap_data = np.zeros((len(learning_rates), len(std_devs)))

# Experiment loop
for i, lr in enumerate(learning_rates):
    for j, std_dev in enumerate(std_devs):
        mlp = MLP(input_size=2, hidden_size=10, output_size=2, learning_rate=lr, batch_size=32, weight_init_std_dev=std_dev)
        mlp.train(X_train, y_train, X_val, y_val, epochs=100, plot_interval=20, enable_plotting=False)

        y_pred = predict(mlp, X_val)
        accuracy = calculate_accuracy(y_val, one_hot_encode(y_pred, num_classes))
        heatmap_data[i, j] = accuracy

        plt.close()

# Plotting heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(ticks=range(len(std_devs)), labels=std_devs)
plt.yticks(ticks=range(len(learning_rates)), labels=learning_rates)
plt.xlabel('Standard Deviation for Weight Initialization')
plt.ylabel('Learning Rate')
plt.title('Validation Accuracy Heatmap')
plt.show()
