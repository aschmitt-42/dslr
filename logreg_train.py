import sys
import json
from utils import read_csv, get_numerical_columns, GetNotesByStudents, sigmoid
from bonus import (
    stochastic_gradient_descent,
    calculate_r2_score,
    calculate_rmse,
    calculate_mae,
    calculate_mape
)



def predict(notesByStudents, weights):
    probabilities = []
    for student in notesByStudents:
        z = weights[0]  # biais
        tuples = zip(weights[1:], student)
        for weight, note in tuples:
            z += weight * note 
        probabilities.append(sigmoid(z)) # sigmoid transforme la somme pondérée en une probabilité entre 0 et 1
    return probabilities


def GradientDescent(notesByStudents, labels, weights, learning_rate=0.1):
    nb_students = len(notesByStudents)
    probabilities = predict(notesByStudents, weights)
    errors = [probabilities[i] - labels[i] for i in range(nb_students)] # erreur entre la probabilité prédite et le label réel pour chaque étudiant

    # mise à jour les poids pour réduire les erreurs
    temp_wj = [0.0] * (len(weights) - 1)
    temp_w0 = sum(errors) / nb_students # le biais moyenne des erreurs

    for j in range(len(weights) - 1):
        for i in range(nb_students):
            temp_wj[j] += errors[i] * notesByStudents[i][j]
        temp_wj[j] /= nb_students

    weights[0] -= learning_rate * temp_w0
    for j in range(len(weights) - 1):
        weights[j + 1] -= learning_rate * temp_wj[j]
    
    return weights

def get_labels(data, house):
    labels = []
    for h in data["Hogwarts House"]:
        if h == house:
            labels.append(1)
        else:
            labels.append(0)
    return labels

def normalize(notesByStudents):
    nb_students = len(notesByStudents)
    nb_features = len(notesByStudents[0])
    normalized  = [[] for _ in range(nb_students)]
    mins = []
    maxs = []

    for i in range(nb_features):
        col_values = [notesByStudents[j][i] for j in range(nb_students)]
        min_val = min(col_values)
        max_val = max(col_values)
        mins.append(min_val)
        maxs.append(max_val)
        for j in range(nb_students):
            if max_val != min_val:
                normalized[j].append((notesByStudents[j][i] - min_val) / (max_val - min_val))
            else:
                normalized[j].append(0)

    return normalized, mins, maxs


def denormalize_weights(weights, mins, maxs):
    """Dénormalise les poids après l'entraînement"""
    w_denorm = [0.0] * len(weights)

    for j in range(len(weights) - 1):
        w_denorm[j + 1] = weights[j + 1] / (maxs[j] - mins[j])

    w_denorm[0] = weights[0]
    for j in range(len(weights) - 1):
        w_denorm[0] -= weights[j + 1] * mins[j] / (maxs[j] - mins[j])

    return w_denorm


def display_metrics(house, labels, predictions):
    
    normalizedNotes, mins, maxs = normalize(notesByStudents)

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    weights_by_house = {}

    for house in houses:
        weights = [0.0] * (len(numerical_cols) + 1)
        labels = get_labels(data, house)

        if use_sgd:
            for _ in range(500):
                weights = stochastic_gradient_descent(
                    normalizedNotes, labels, weights,
                    learning_rate=0.1, epochs=1)
        else:
            for _ in range(500):
                weights = GradientDescent(
                    normalizedNotes, labels, weights,
                    learning_rate=0.1)

        weights_by_house[house] = weights

        predictions = predict(normalizedNotes, weights)
        display_metrics(house, labels, predictions)

    model = {"weights": weights_by_house, "min": mins, "max": maxs}
    try:
        with open("weights.json", "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)
    except Exception as e:
        print(f"Probleme avec le json : {e}")
        exit(1)


if __name__ == "__main__":
    main()