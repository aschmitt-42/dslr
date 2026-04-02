import sys
import math
from utils import read_csv, get_numerical_columns, ListEachNotesByHouse, get_data_by_column, calculateMean

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(notesByStudents, weights):
    proba = []
    for student in notesByStudents:
        z = weights[0]  # biais
        tuples = zip(weights[1:], student)
        for w, x in tuples:
            z += w * x
        proba.append(sigmoid(z))
    return proba

def GradientDescent(notesByStudents, labels, weights, learning_rate=0.1):
    n = len(notesByStudents)
    proba = predict(notesByStudents, weights)
    errors = [proba[i] - labels[i] for i in range(n)]
    
    gradient_w0 = sum(errors) / n 

    gradient_wj = [0.0] * (len(weights) - 1)
    for j in range(len(weights) - 1):
        for i in range(n):
            gradient_wj[j] += errors[i] * notesByStudents[i][j]
        gradient_wj[j] /= n

    weights[0] -= learning_rate * gradient_w0
    for j in range(len(weights) - 1):
        weights[j + 1] -= learning_rate * gradient_wj[j]
    
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

    for i in range(nb_features):
        col_values = [notesByStudents[j][i] for j in range(nb_students)]
        min_val = min(col_values)
        max_val = max(col_values)
        for j in range(nb_students):
            if max_val != min_val:
                normalized[j].append((notesByStudents[j][i] - min_val) / (max_val - min_val))
            else:
                normalized[j].append(0)

    return normalized

def get_means(data, numerical_cols):
    means = {}
    for col in numerical_cols:
        values = get_data_by_column(data, col)
        means[col] = calculateMean(values)
    return means

def GetNotesByStudents(data, numerical_cols, means):
    nbStudents = len(data["Index"])
    notes  = [[] for _ in range(nbStudents)]

    for i in range(nbStudents):
        for col in numerical_cols:
            if (data[col][i] == '' or data[col][i] is None):
                notes[i].append(means[col])
            else:
                notes[i].append(float(data[col][i]))
    return notes

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset>")
        return
    header, data = read_csv(sys.argv[1])
    numerical_cols = get_numerical_columns(header)
    means = get_means(data, numerical_cols)
    notesByStudents = GetNotesByStudents(data, numerical_cols, means)
    
    normalizedNotes = normalize(notesByStudents)

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    for house in houses:
        weights = [0.0] * (len(numerical_cols) + 1)
        labels = get_labels(data, house)
        for _ in range(1000):
            weights = GradientDescent(normalizedNotes, labels, weights, learning_rate=0.1)
        # print(f"Weights for {house}: {weights}")
        predictions = predict(normalizedNotes, weights)
        correct = sum(1 for p, l in zip(predictions, labels) if (p >= 0.5) == l)
        accuracy = correct / len(labels) * 100
        print(f"{house}: {accuracy:.2f}%")
  
    

if __name__ == "__main__":
    main()