import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlp_model_3 import MLP  # Make sure to import the correct model
from ml_utils import predict, calculate_accuracy

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

# One-hot encode and normalize
y_train = one_hot_encode(y_train, num_classes)
y_val = one_hot_encode(y_val, num_classes)
X_train = normalize_features(X_train)
X_val = normalize_features(X_val)

# Select the best and worst performing models based on your earlier heatmap results
best_lr = 0.0001  
best_std_dev = 0.7

worst_lr = 0.1  
worst_std_dev = 0.9  

# Function to plot activations for each layer
def plot_activations(activations_tuple, title):
    # activations_tuple contains a tuple of activations for each point in time (initial, halfway, final)
    for activation_time, activations in zip(["Initial", "Halfway", "Final"], activations_tuple):
        # Each 'activations' is a tuple with the activations for each layer
        for layer_idx, layer_activations in enumerate(activations):
            # Transpose the activations to get shape (num_samples, num_neurons)
            layer_activations = layer_activations.T
            plt.figure(figsize=(12, 6))
            sns.heatmap(layer_activations, cmap='viridis')
            plt.title(f'{title} - {activation_time} Training - Layer {layer_idx+1}')
            plt.xlabel('Neuron ID')
            plt.ylabel('Data Point ID')
            plt.show()

# Train and visualize activations for the best model
best_model = MLP(learning_rate=best_lr, weight_init_std_dev=best_std_dev)  
train_losses, val_losses, (initial_activations, halfway_activations, final_activations) = best_model.train(
    X_train, y_train, X_val, y_val, epochs=100, enable_plotting=False
)
best_model_activations = (initial_activations, halfway_activations, final_activations)
plot_activations(best_model_activations, "Best Model")

# Repeat for the worst model
worst_model = MLP(learning_rate=worst_lr, weight_init_std_dev=worst_std_dev)  
train_losses, val_losses, (initial_activations, halfway_activations, final_activations) = worst_model.train(
    X_train, y_train, X_val, y_val, epochs=100, enable_plotting=False
)
worst_model_activations = (initial_activations, halfway_activations, final_activations)
plot_activations(worst_model_activations, "Worst Model")
