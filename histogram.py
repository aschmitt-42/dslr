import sys
from utils import read_csv, get_numerical_columns, calculateMean, calculateStandardDeviation, ListEachNotesByHouse

import matplotlib.pyplot as plt


def homogeneity_score(NotesByHouse, col_name):
    list_means = []
    for house, notes in NotesByHouse.items():
        mean = calculateMean(notes)
        list_means.append(mean)

    mean_of_means = calculateMean(list_means)
    std = calculateStandardDeviation(list_means, mean_of_means)

    return std / abs(mean_of_means) if mean_of_means != 0 else 0
    
def displayedAll(data, numerical_cols, ColorsMaisons):
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))    # creation de 16 shema invisible
    for j in range(len(numerical_cols), 16):
        axes[j // 4][j % 4].set_visible(False)

    homogenity_scores = {}
    for i, col in enumerate(numerical_cols):            # pour chaque colone numerique, on affiche un histogramme
        ax = axes[i // 4][i % 4]

        NotesByHouse = ListEachNotesByHouse(data, col)  # on recupere les notes de chaque maison pour la colone en cours
        for house, color in ColorsMaisons.items():      # pour chaque maison, on affiche un histogramme de ses notes avec une couleur differente
            values = NotesByHouse[house]
            ax.hist(values, bins=20, alpha=0.5, label=house, color=color)
        
        homogenity_scores[col] = homogeneity_score(NotesByHouse, col)              # on affiche le score d'homogeneite pour la colone en cours
        ax.set_title(col, fontsize=9)
        ax.set_ylabel('student count')
        ax.legend(fontsize=7)
        ax.grid()
        plt.tight_layout(pad=3.0)    
    plt.savefig("histograms_all.png")


def displayedBest(data, numerical_cols, ColorsMaisons):
    homogenity_scores = {}
    for _, col in enumerate(numerical_cols):            # pour chaque colone numerique, on affiche un histogramme
        NotesByHouse = ListEachNotesByHouse(data, col)  # on recupere les notes de chaque maison pour la colone en cours
        homogenity_scores[col] = homogeneity_score(NotesByHouse, col)
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
    plt.savefig("histogram.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 histogram.py <dataset>")
        return
    all = False
    if len(sys.argv) == 3 and sys.argv[2] == "--all":
        all = True
    header, data = read_csv(sys.argv[1])
    numerical_cols = get_numerical_columns(header)

    ColorsMaisons = {
        "Ravenclaw": "blue",
        "Slytherin": "green",
        "Gryffindor": "red",
        "Hufflepuff": "yellow"
    }

    if (all):
        displayedAll(data, numerical_cols, ColorsMaisons)
    else:
        displayedBest(data, numerical_cols, ColorsMaisons)
    

if __name__ == "__main__":
    main()