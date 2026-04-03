import sys
import math
import json
from utils import read_csv, get_numerical_columns, GetNotesByStudents, sigmoid



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


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset>")
        return

    header, data = read_csv(sys.argv[1])
    numerical_cols = get_numerical_columns(header)
    notesByStudents = GetNotesByStudents(data, numerical_cols) # Chaque element de la liste correspond a un etudiant et contient la liste de ses notes
    
    # normalisation des notes entre 0 et 1, évite que certaines features dominent et accélère la convergence du modèle
    normalizedNotes, mins, maxs = normalize(notesByStudents)

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    weights_by_house = {}

    for house in houses:
        weights = [0.0] * (len(numerical_cols) + 1) # Initialisation des poids a 0
        labels = get_labels(data, house)            # labels indique si etudiant appartient a la maison
        for _ in range(500):                       # Entraînement du modèle de régression logistique
            weights = GradientDescent(normalizedNotes, labels, weights, learning_rate=0.1)
        weights_by_house[house] = weights
        predictions = predict(normalizedNotes, weights)
        correct = sum(1 for p, l in zip(predictions, labels) if (p >= 0.5) == l)
        accuracy = correct / len(labels) * 100
        print(f"{house}: {accuracy:.2f}%")

    model = { "weights": weights_by_house, "min": mins, "max": maxs }
    try:
        with open(f"weights.json", "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)
    except Exception as e:
        print(f"Probleme avec le json : {e}")
        exit(1)
        
        


  
    

if __name__ == "__main__":
    main()