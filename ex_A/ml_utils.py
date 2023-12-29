import numpy as np

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def predict(model, X):
    outputs = model.forward(X)
    print("Shape of softmax outputs:", outputs.shape)
    print("Softmax output for a sample:", outputs[:, :5])  # Print first few probabilities for checking
    predictions = np.argmax(outputs, axis=0)  # Convert softmax probabilities to class labels
    return predictions