import sys
import matplotlib.pyplot as plt
from utils import read_csv, get_numerical_columns, ListEachNotesByHouse

def ListEachNotesByHouseWithTwoColumns(data, col_name1, col_name2):
    Houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    NotesByHouse_x = {house: [] for house in Houses}
    NotesByHouse_y = {house: [] for house in Houses}

    for i in range(len(data['Hogwarts House'])):
        if data[col_name1][i] == '' or data[col_name1][i] is None or data[col_name2][i] == '' or data[col_name2][i] is None:
            continue
        house = data['Hogwarts House'][i]
        if house in Houses:
            NotesByHouse_x[house].append(float(data[col_name1][i]))
            NotesByHouse_y[house].append(float(data[col_name2][i]))

    return NotesByHouse_x, NotesByHouse_y

def histo(data, col_name, ColorsMaisons, ax):
    NotesByHouse = ListEachNotesByHouse(data, col_name)  # on recupere les notes de chaque maison pour la colone en cours
    for house, color in ColorsMaisons.items():          # pour chaque maison, on affiche un histogramme de ses notes avec une couleur differente
        values = NotesByHouse[house]
        ax.hist(values, bins=20, alpha=0.5, label=house, color=color)
    
    ax.grid()

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
    len_numerical_cols = len(numerical_cols)
    fig, axes = plt.subplots(len_numerical_cols, len_numerical_cols, figsize=(20, 15))

    for i in range(len_numerical_cols):
        for j in range(len_numerical_cols):
            if (i < j):
                axes[i][j].set_visible(False)
                continue
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if (i == len_numerical_cols - 1):
                ax.set_xlabel(numerical_cols[j], fontsize=7)
            if (j == 0):
                ax.set_ylabel(numerical_cols[i], fontsize=7)

            if (i == j):
                histo(data, numerical_cols[i], ColorsMaisons, ax)
                continue

            NotesByHouse_x, NotesByHouse_y = ListEachNotesByHouseWithTwoColumns(data, numerical_cols[i], numerical_cols[j])

            for house, color in ColorsMaisons.items():
                ax.scatter(NotesByHouse_x[house], NotesByHouse_y[house], alpha=0.5, label=house, color=color, s=1)

            ax.grid()

    plt.tight_layout(pad=1.5)
    plt.savefig("pair_plot.png")


if __name__ == "__main__":
    main()