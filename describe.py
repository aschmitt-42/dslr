import sys
from utils import (
    read_csv, 
    get_numerical_columns, 
    get_data_by_column, 
    calculateMean,
    calculateStandardDeviation,
    findMin,
    findMax,
    calculatePercentile
)


def get_count(data, col_name):
    #Compte le nombre de valeurs non-vides pour une colonne
    count = 0
    for val in data[col_name]:
        if val != '' and val is not None:
            try:
                float(val)
                count += 1
            except ValueError:
                continue
    return count


def describe_column(data, col_name):
    #Calcule tous les statistiques pour une colonne
    values = get_data_by_column(data, col_name)
    
    if len(values) == 0:
        return None
    
    mean = calculateMean(values)
    std = calculateStandardDeviation(values, mean)
    min_val = findMin(values)
    max_val = findMax(values)
    
    percentile_25 = calculatePercentile(values, 25)
    percentile_50 = calculatePercentile(values, 50)
    percentile_75 = calculatePercentile(values, 75)
    count = get_count(data, col_name)
    
    return {
        'Count': count,
        'Mean': mean,
        'Std': std,
        'Min': min_val,
        '25%': percentile_25,
        '50%': percentile_50,
        '75%': percentile_75,
        'Max': max_val,
    }


def print_table(data, numerical_cols):
    #Affiche les statistiques sous forme de tableau
    
    # Collecte les statistiques pour chaque colonne
    stats_by_col = {}
    for col in numerical_cols:
        stats_by_col[col] = describe_column(data, col)
    
    # Définit l'ordre des statistiques
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    
    # Calcule la largeur optimale pour chaque colonne
    col_widths = {}
    for col in numerical_cols:
        # Commence avec la largeur du nom de la colonne
        width = len(col) + 2
        # Vérifie que c'est assez pour les nombres (14 caractères pour "1234567.890123")
        width = max(width, 14)
        col_widths[col] = width
    
    # Largeur pour la colonne des labels (statistiques)
    label_width = 10
    
    # Affiche l'en-tête (noms des colonnes)
    print(f"{'':<{label_width}}", end='')
    for col in numerical_cols:
        print(f"{col:>{col_widths[col]}}", end='')
    print()
    
    # Affiche chaque ligne de statistiques
    for stat in stat_names:
        print(f"{stat:<{label_width}}", end='')
        for col in numerical_cols:
            if stats_by_col[col] is not None:
                value = stats_by_col[col][stat]
                if value is not None:
                    print(f"{value:>{col_widths[col]}.6f}", end='')
                else:
                    print(f"{'N/A':>{col_widths[col]}}", end='')
            else:
                print(f"{'N/A':>{col_widths[col]}}", end='')
        print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset>")
        return

    header, data = read_csv(sys.argv[1])
    
    if data is None:
        return
    
    numerical_cols = get_numerical_columns(header)
    
    if not numerical_cols:
        print("Aucune colonne numérique trouvée dans le dataset")
        return
    
    print_table(data, numerical_cols)


if __name__ == "__main__":
    main()
