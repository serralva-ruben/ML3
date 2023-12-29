import numpy as np
#function to calculate the accuracy
def calculate_accuracy(y_true, y_pred):
    if y_true.ndim > 1:  # Check if y_true is one-hot encoded
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:  # Check if y_pred is one-hot encoded
        y_pred = np.argmax(y_pred, axis=1)
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy
#function to compute the confusion matrix
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])
#function to use the model to make a prediction using its forward function
def predict(model, X):
    return np.argmax(model.forward(X), axis=0)
