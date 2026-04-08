import sys
import matplotlib.pyplot as plt
from pair_plot import ListEachNotesByHouseWithTwoColumns
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

    return best_pair, max_score

def displayedAllPairs(data, numerical_cols, ColorsMaisons):
    len_numerical_cols = len(numerical_cols)
    for i in range(len_numerical_cols):
        for j in range(i + 1, len_numerical_cols):
            col_x = numerical_cols[i]
            col_y = numerical_cols[j]

            # calcul du score de Pearson pour la paire de features (col_x, col_y)
            list_note_x = get_data_by_column_with_none(data, col_x)
            list_note_y = get_data_by_column_with_none(data, col_y)
            score = calculatePearson(list_note_x, list_note_y)
            
            # affichage du scatter plot pour la paire de features (col_x, col_y)
            NotesByHouse_x, NotesByHouse_y = ListEachNotesByHouseWithTwoColumns(data, col_x, col_y)
            fig, ax = plt.subplots(figsize=(10, 7))
            for house, color in ColorsMaisons.items():
                ax.scatter(NotesByHouse_x[house], NotesByHouse_y[house], alpha=0.5, label=house, color=color, s=10)

            # configuration du graphique
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.set_title(f"{col_x} vs {col_y} (Pearson={score:.4f})")
            ax.legend()
            ax.grid()
            plt.tight_layout()
            plt.show()
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scatter_plot.py <dataset> [--all]")
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
        displayedAllPairs(data, numerical_cols, ColorsMaisons)
        return
    
    best_pair, score = max_abs_pearson(data, numerical_cols)
    col_x, col_y = best_pair

    NotesByHouse_x, NotesByHouse_y = ListEachNotesByHouseWithTwoColumns(data, col_x, col_y)

    plt.figure(figsize=(10, 7))
    for house, color in ColorsMaisons.items():
        plt.scatter(NotesByHouse_x[house], NotesByHouse_y[house], alpha=0.5, label=house, color=color, s=10)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f"Most similar features: {col_x} vs {col_y} (Pearson={score:.4f})")
    plt.legend()
    plt.grid()
    plt.tight_layout(pad=3.0)                           # pour eviter que les titres et les axes se chevauchent
    plt.savefig("scatter_plot.png")

if __name__ == "__main__":
    main()