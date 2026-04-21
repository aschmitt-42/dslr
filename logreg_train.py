import sys
import csv
from sklearn.metrics import accuracy_score
from utils import read_csv, get_columns_for_gradient, GetNotesByStudents, sigmoid
from bonus import (
    stochastic_gradient_descent,
    mini_batch_gradient_descent,
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

    temp_wj = [0.0] * (len(weights) - 1)
    temp_w0 = sum(errors) / nb_students

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

def save_weights(weights_by_house, filename, numerical_cols):
    header = ["Hogwarts House", "Biais"] + numerical_cols
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for house, weights in weights_by_house.items():
                row = [house] + weights
                writer.writerow(row)
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier CSV : {e}")
        exit(1)

def display_metrics(house, labels, predictions):
    r2 = calculate_r2_score(labels, predictions)
    rmse = calculate_rmse(labels, predictions)
    mae = calculate_mae(labels, predictions)
    mape = calculate_mape(labels, predictions)
    
    # Convert predictions to binary (0 or 1) for accuracy score
    binary_predictions = [1 if p >= 0.5 else 0 for p in predictions]
    accuracy = accuracy_score(labels, binary_predictions) * 100
    
    print(f"{house}: Accuracy = {accuracy:.4f}%, R² Score = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}%")


def main():
    use_sgd = False
    use_mini_batch = False
    
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 logreg_train.py <dataset.csv> [--stochastic | --mini-batch]")
        exit(1)
    if len(sys.argv) > 2 and sys.argv[2] == "--stochastic":
        use_sgd = True
    elif len(sys.argv) > 2 and sys.argv[2] == "--mini-batch":
        use_mini_batch = True
    else:
        print("Usage: python3 logreg_train.py <dataset.csv> [--stochastic | --mini-batch]")
        exit(1)

    header, data = read_csv(sys.argv[1])
    numerical_cols = get_columns_for_gradient()

    notesByStudents = GetNotesByStudents(data, numerical_cols)
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
        elif use_mini_batch:
            for _ in range(500):
                weights = mini_batch_gradient_descent(
                    normalizedNotes, labels, weights,
                    batch_size=32, learning_rate=0.1, epochs=1)
        else:
            for _ in range(500):
                weights = GradientDescent(
                    normalizedNotes, labels, weights,
                    learning_rate=0.1)


        predictions = predict(normalizedNotes, weights)
        display_metrics(house, labels, predictions)
        weights_denorm = denormalize_weights(weights, mins, maxs)
        weights_by_house[house] = weights_denorm
        save_weights(weights_by_house, "weights.csv", numerical_cols)   


if __name__ == "__main__":
    main()
