import sys
from utils import read_csv, get_numerical_columns, calculateMean, calculateStandardDeviation, ListEachNotesByHouse

import matplotlib.pyplot as plt


def homogeneity_score(NotesByHouse):
    list_means = []
    for house, notes in NotesByHouse.items():
        mean = calculateMean(notes)
        list_means.append(mean)

    mean_of_means = calculateMean(list_means)
    std = calculateStandardDeviation(list_means, mean_of_means)

    return std / abs(mean_of_means) if mean_of_means != 0 else 0


def displayedBest(data, numerical_cols, ColorsMaisons):
    homogenity_scores = {}
    for _, col in enumerate(numerical_cols):            # pour chaque colone numerique, on affiche un histogramme
        NotesByHouse = ListEachNotesByHouse(data, col)  # on recupere les notes de chaque maison pour la colone en cours
        homogenity_scores[col] = homogeneity_score(NotesByHouse)
    best = min(homogenity_scores, key=homogenity_scores.get)
    
    plt.figure(figsize=(10, 7))
    
    NotesByHouse = ListEachNotesByHouse(data, best)
    for house, color in ColorsMaisons.items():          # pour chaque maison, on affiche un histogramme de ses notes avec une couleur differente
            values = NotesByHouse[house]
            plt.hist(values, bins=20, alpha=0.5, label=house, color=color)

    plt.xlabel(best)
    plt.ylabel("Number of Students")
    plt.title(f"Most homogenous: {best} (Pearson={homogenity_scores[best]:.4f})")

    plt.tight_layout(pad=3.0)                           # pour eviter que les titres et les axes se chevauchent
    plt.savefig("output/histogram.png")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py <dataset>")
        return
    header, data = read_csv(sys.argv[1])
    numerical_cols = get_numerical_columns(header)

    ColorsMaisons = {
        "Ravenclaw": "blue",
        "Slytherin": "green",
        "Gryffindor": "red",
        "Hufflepuff": "yellow"
    }

    displayedBest(data, numerical_cols, ColorsMaisons)
    

if __name__ == "__main__":
    main()