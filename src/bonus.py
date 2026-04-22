import math
import random
from utils import sigmoid, calculateMean


def stochastic_gradient_descent(notesByStudents, labels, weights, learning_rate, epochs):
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


def mini_batch_gradient_descent(notesByStudents, labels, weights, batch_size, learning_rate, epochs):
    nb_students = len(notesByStudents)
    
    if batch_size > nb_students:
        batch_size = nb_students
    
    for epoch in range(epochs):
        # Shuffle les indices
        indices = list(range(nb_students))
        for i in range(nb_students - 1, 0, -1):
            j = int(random.random() * (i + 1))
            indices[i], indices[j] = indices[j], indices[i]
        
        for batch_start in range(0, nb_students, batch_size):
            batch_end = min(batch_start + batch_size, nb_students)
            batch_indices = indices[batch_start:batch_end]
            
            # Calcul du gradient sur le batch
            gradient_w0 = 0.0
            gradient_wj = [0.0] * (len(weights) - 1)
            
            for idx in batch_indices:
                z = weights[0]
                for j in range(len(notesByStudents[idx])):
                    z += weights[j + 1] * notesByStudents[idx][j]
                
                pred = sigmoid(z)
                error = pred - labels[idx]
                
                gradient_w0 += error
                for j in range(len(notesByStudents[idx])):
                    gradient_wj[j] += error * notesByStudents[idx][j]
            
            # Moyenne du gradient sur le batch
            batch_size_actual = batch_end - batch_start
            gradient_w0 /= batch_size_actual
            for j in range(len(gradient_wj)):
                gradient_wj[j] /= batch_size_actual
            
            weights[0] -= learning_rate * gradient_w0
            for j in range(len(weights) - 1):
                weights[j + 1] -= learning_rate * gradient_wj[j]
    
    return weights

def calculate_variance(values):
    if len(values) == 0:
        return None
    mean = calculateMean(values)
    if mean is None:
        return None
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance


def calculate_skewness(values):
    if len(values) < 2:
        return None
    mean = calculateMean(values)
    variance = calculate_variance(values)
    if mean is None or variance is None or variance == 0:
        return 0.0
    std = math.sqrt(variance)
    third_moment = sum((x - mean) ** 3 for x in values) / len(values)
    skewness = third_moment / (std ** 3)
    return skewness


def calculate_kurtosis(values):
    if len(values) < 4:
        return None
    mean = calculateMean(values)
    variance = calculate_variance(values)
    if mean is None or variance is None or variance == 0:
        return 0.0
    std = math.sqrt(variance)
    fourth_moment = sum((x - mean) ** 4 for x in values) / len(values)
    kurtosis_excess = (fourth_moment / (std ** 4)) - 3
    return kurtosis_excess 