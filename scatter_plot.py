import sys
import matplotlib.pyplot as plt
from utils import read_csv, get_data_by_column, get_numerical_columns, calculateMean, calculateStandardDeviation, ListEachNotesByHouse, get_data_by_column_with_none

def delete_None(values_x, values_y):
    new_values_x = []
    new_values_y = []
    for i in range(len(values_x)):
        if values_x[i] == None or values_y[i] == None:
            continue
        new_values_x.append(values_x[i])
        new_values_y.append(values_y[i])
    return new_values_x, new_values_y

def calculatePearson(values_x, values_y):
    raw_x, raw_y = delete_None(values_x, values_y)  # on supprime les 2 valeurs pour chaque None trouver dans les listes

    if (len(raw_x) != len(raw_y)):
        print("Error: The lengths of the two lists are not equal.")
        return 0

    mean_x = calculateMean(raw_x)
    mean_y = calculateMean(raw_y)
    std_x = calculateStandardDeviation(raw_x, mean_x)
    std_y = calculateStandardDeviation(raw_y, mean_y)

    numerator = 0
    for i in range(len(raw_x)):
        numerator += (raw_x[i] - mean_x) * (raw_y[i] - mean_y)
    
    if std_x > 0 and std_y > 0:
        return numerator / (len(raw_x) * std_x * std_y)
    return 0

def max_abs_pearson(data, numerical_cols):
    max_score = 0
    best_pair = (None, None)

    len_numerical_cols = len(numerical_cols)
    for i in range(len_numerical_cols):
        for j in range(i + 1, len_numerical_cols):

            list_note_x = get_data_by_column_with_none(data, numerical_cols[i])
            list_note_y = get_data_by_column_with_none(data, numerical_cols[j])

            pearson_score = calculatePearson(list_note_x, list_note_y)

            if abs(pearson_score) > abs(max_score):
                max_score = pearson_score
                best_pair = (numerical_cols[i], numerical_cols[j])

    print(f"\nPaire de features la plus corrélée: {best_pair[0]} et {best_pair[1]} avec un score de Pearson de {max_score:.4f}")
    
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scatter_plot.py <dataset>")
        return
    header, data = read_csv(sys.argv[1])
    numerical_cols = get_numerical_columns(header)

    ColorsMaisons = {
        "Ravenclaw": "blue",
        "Slytherin": "green",
        "Gryffindor": "red",
        "Hufflepuff": "yellow"
    }

    fig, axes = plt.subplots(4, 4, figsize=(20, 15))    # creation de 16 shema invisible
    for j in range(len(numerical_cols), 16):
        axes[j // 4][j % 4].set_visible(False)

    for i, col in enumerate(numerical_cols):            # pour chaque colone numerique, on affiche un histogramme
        ax = axes[i // 4][i % 4]

        NotesByHouse = ListEachNotesByHouse(data, col)  # on recupere les notes de chaque maison pour la colone en cours
        for house, color in ColorsMaisons.items():      # pour chaque maison, on affiche un histogramme de ses notes avec une couleur differente
            values = NotesByHouse[house]
            ax.scatter(range(len(values)), values, alpha=0.5, label=house, color=color, s=1)
        
        ax.set_title(col, fontsize=9)
        ax.set_ylabel('student count')
        ax.legend(fontsize=7)
        ax.grid()

    plt.tight_layout(pad=3.0)                           # pour eviter que les titres et les axes se chevauchent
    plt.savefig("scatter_plot.png")

    max_abs_pearson(data, numerical_cols)               # on affiche la paire de features la plus corrélée selon le score de Pearson

if __name__ == "__main__":
    main()