import sys
import csv
from logreg_train import GetNotesByStudents
from utils import get_columns_for_guardian, read_csv, sigmoid


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
        with open(filename, "r", newline='') as f:
            reader = csv.DictReader(f)
            weights_by_house = {}
            for row in reader:
                house = row["Hogwarts House"]
                weights = [float(row["Biais"])]
                for col in get_columns_for_guardian():
                    weights.append(float(row[col]))
                weights_by_house[house] = weights
        return weights_by_house
    except Exception as e:
        print(f"Probleme avec le csv : {e}")
        exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 logreg_predict.py <dataset> <weights>")
        return

    weights_by_house = load_weights(sys.argv[2])
    header, data = read_csv(sys.argv[1])
    numerical_cols = get_columns_for_guardian()
    notesByStudents = GetNotesByStudents(data, numerical_cols) # Chaque element de la liste correspond a un etudiant et contient la liste de ses notes
    
    
    predicted_houses = []
    for i, student in enumerate(notesByStudents):
        scores = {}
        for house, weights in weights_by_house.items():
            score = weights[0] 
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