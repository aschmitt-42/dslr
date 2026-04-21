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


def mini_batch_gradient_descent(notesByStudents, labels, weights, batch_size=32, learning_rate=0.1, epochs=500):
    """
    Mini-Batch Gradient Descent : compromise entre BGD et SGD.
    Traite des petits groupes de samples à la fois.
    
    Args:
        notesByStudents: Liste de listes (features pour chaque étudiant)
        labels: Liste de labels (0 ou 1)
        weights: Poids du modèle [biais, w1, w2, ..., wn]
        batch_size: Taille des mini-batches (défaut: 32)
        learning_rate: Taux d'apprentissage (défaut: 0.1)
        epochs: Nombre d'epochs (défaut: 500)
    
    Returns:
        weights: Poids mis à jour après l'entraînement
    """
    nb_students = len(notesByStudents)
    
    # Ajuste batch_size si plus grand que le dataset
    if batch_size > nb_students:
        batch_size = nb_students
    
    for epoch in range(epochs):
        # Shuffle les indices
        indices = list(range(nb_students))
        for i in range(nb_students - 1, 0, -1):
            j = int(random.random() * (i + 1))
            indices[i], indices[j] = indices[j], indices[i]
        
        # Traite par batches
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
            
            # Mise à jour des poids
            weights[0] -= learning_rate * gradient_w0
            for j in range(len(weights) - 1):
                weights[j + 1] -= learning_rate * gradient_wj[j]
    
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

