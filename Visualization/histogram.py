import pandas as pd
from utils import read_csv, get_data_by_column, get_numerical_columns

import matplotlib.pyplot as plt

def ListEachNotesByHouse(data, col_name):
    Houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    NotesByHouse = {house: [] for house in Houses}

    for i in range(len(data['Hogwarts House'])):
        if data[col_name][i] == '' or data[col_name][i] is None:
            continue
        house = data['Hogwarts House'][i]
        if house in Houses:
            NotesByHouse[house].append(float(data[col_name][i]))

    return NotesByHouse


def main():
    header, data = read_csv('datasets/dataset_train.csv')
    numerical_cols = get_numerical_columns(header)

    ColorsMaisons = {
        "Ravenclaw": "blue",
        "Slytherin": "green",
        "Gryffindor": "red",
        "Hufflepuff": "yellow"
    }

    fig, axes = plt.subplots(4, 4, figsize=(20, 15)) # 10, 8
    for j in range(len(numerical_cols), 16):
        axes[j // 4][j % 4].set_visible(False)

    student_count = len(data['Hogwarts House'])
    for i, col in enumerate(numerical_cols):
        ax = axes[i // 4][i % 4]

        NotesByHouse = ListEachNotesByHouse(data, col)
        for house, color in ColorsMaisons.items():
            values = NotesByHouse[house]
            ax.hist(values, bins=20, alpha=0.5, label=house, color=color)

        ax.set_title(col, fontsize=9)
        ax.set_ylabel('student count')
        ax.legend(fontsize=7)
        ax.grid()

    plt.tight_layout(pad=3.0)
    plt.show()



if __name__ == "__main__":
    main()