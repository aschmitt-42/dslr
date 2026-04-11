import math
import random
from utils import sigmoid


def stochastic_gradient_descent(notesByStudents, labels, weights, learning_rate=0.1, epochs=500):
    nb_students = len(notesByStudents)
    
    for epoch in range(epochs):
        indices = list(range(nb_students))
        for i in range(nb_students - 1, 0, -1):
            j = int(random.random() * (i + 1))
            indices[i], indices[j] = indices[j], indices[i]
        
        for idx in indices:
            z = weights[0]
            for j in range(len(notesByStudents[idx])):
                z += weights[j + 1] * notesByStudents[idx][j]
            pred = sigmoid(z)
            
            error = pred - labels[idx]
            
            weights[0] -= learning_rate * error
            for j in range(len(notesByStudents[idx])):
                weights[j + 1] -= learning_rate * error * notesByStudents[idx][j]
    
    return weights

def calculate_r2_score(y_true, y_pred):
    if len(y_true) == 0:
        return None
    
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    ss_tot = sum((y_true[i] - mean_y) ** 2 for i in range(len(y_true)))
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def calculate_rmse(y_true, y_pred):
    if len(y_true) == 0:
        return None
    
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)
    return math.sqrt(mse)


def calculate_mae(y_true, y_pred):
    if len(y_true) == 0:
        return None
    
    return sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true))) / len(y_true)


def calculate_mape(y_true, y_pred):
    if len(y_true) == 0:
        return None
    
    total_error = 0
    count = 0
    for i in range(len(y_true)):
        if y_true[i] != 0:
            total_error += abs(y_true[i] - y_pred[i]) / abs(y_true[i])
            count += 1
    
    if count == 0:
        return None
    
    return (total_error / count) * 100

