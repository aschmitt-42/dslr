import json
import sys
import csv


from logreg_train import GetNotesByStudents
from utils import get_numerical_columns, read_csv, sigmoid


def normalizeNotes(notesByStudents, mins, maxs):
    nb_students = len(notesByStudents)
    nb_features = len(notesByStudents[0])
    normalized  = [[] for _ in range(nb_students)]

    for i in range(nb_features):
        for j in range(nb_students):
            if mins[i] != maxs[i]:
                normalized[j].append((notesByStudents[j][i] - mins[i]) / (maxs[i] - mins[i]))
            else:
                normalized[j].append(0)

    return normalized

def load_weights(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            model = json.load(f)
            return model["weights"], model["min"], model["max"]
    except Exception as e:
        print(f"Probleme avec le json : {e}")
        exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 logreg_predict.py <dataset> <weights>")
        return

    weights_by_house, mins, maxs = load_weights(sys.argv[2])
    header, data = read_csv(sys.argv[1])
    numerical_cols = get_numerical_columns(header)
    notesByStudents = GetNotesByStudents(data, numerical_cols) # Chaque element de la liste correspond a un etudiant et contient la liste de ses notes
    
    # normalisation des notes entre 0 et 1, évite que certaines features dominent et accélère la convergence du modèle
    normalizedNotes = normalizeNotes(notesByStudents, mins, maxs)
    predicted_houses = []
    for i, student in enumerate(normalizedNotes):
        scores = {}
        for house, weights in weights_by_house.items():
            score = weights[0] # bias
            for j in range(len(student)):
                score += student[j] * weights[j + 1]
            scores[house] = sigmoid(score)
        
        predicted_house = max(scores, key=scores.get)
        predicted_houses.append(predicted_house)

    with open("houses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Hogwarts House"])
        for i, house in enumerate(predicted_houses):
            writer.writerow([i, house])


if __name__ == "__main__":
    main()