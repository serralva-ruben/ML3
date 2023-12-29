import pandas as pd
import numpy as np
from mlp_model import MLP
from ml_utils import confusion_matrix, predict, calculate_accuracy
import matplotlib.pyplot as plt
#enable iterative mode for plotlib
plt.ion()

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

#function to load and preprocess the dataset xlsx files
def load_data(filename):
    df = pd.read_excel(filename)
    X = df[['X_0', 'X_1']].values
    y = df['y'].values
    return X, y

# functin to normalize the data
def normalize_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# We load the inputs and outputs for both training and validation datasets
X_train, y_train = load_data('THA3train.xlsx')
X_val, y_val = load_data('THA3validate.xlsx')
num_classes = np.max(y_train) + 1

y_train = one_hot_encode(y_train, num_classes)
y_val = one_hot_encode(y_val, num_classes)

# we normalize the data using the function above
X_train = normalize_features(X_train)
X_val = normalize_features(X_val)

# We choose our parameters for our model, the input, hidden and output size are the ones asked in the homework
# For the learning rate, batch size and epochs we chosed something that seemed reasonable, the learning rate of 0.01 works but doesn't overshoot
input_size = 2
hidden_size = 10
output_size = 2
learning_rate = 0.01
batch_size = 32
epochs = 100

#we initialize our model
mlp = MLP(input_size, hidden_size, output_size, learning_rate, batch_size)
train_losses, val_losses = mlp.train(X_train, y_train, X_val, y_val, epochs, plot_interval=5)

plt.figure(figsize=(8, 6))
plt.title('Training and Validation Loss Over Epochs')
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.legend()
plt.show()

#we use the model to predict the classes so that we can later calculate the accuracy and the confusion matrix
y_pred = predict(mlp, X_val)
#then using these helper functions we calculate the accuracy and the confusion matrix and we print them
accuracy = calculate_accuracy(y_val, one_hot_encode(y_pred, num_classes))
conf_matrix = confusion_matrix(y_val, one_hot_encode(y_pred, num_classes))
print(f'Validation Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix: \n{conf_matrix}')

plt.show(block=True)

