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
from bonus import (
    calculate_variance,
    calculate_skewness,
    calculate_kurtosis
)


def get_count(data, col_name):
    count = 0
    for val in data[col_name]:
        if val != '' and val is not None:
            try:
                float(val)
                count += 1
            except ValueError:
                continue
    return count


def describe_column(data, col_name, show_bonus=False):
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
    
    stats = {
        'Count': count,
        'Mean': mean,
        'Std': std,
        'Min': min_val,
        '25%': percentile_25,
        '50%': percentile_50,
        '75%': percentile_75,
        'Max': max_val,
    }
    
    if show_bonus:
        variance = calculate_variance(values)
        skewness = calculate_skewness(values)
        kurtosis = calculate_kurtosis(values)
        stats['Variance'] = variance
        stats['Skewness'] = skewness
        stats['Kurtosis'] = kurtosis
    
    return stats


def print_table(data, numerical_cols, show_bonus=False):
    stats_by_col = {}
    for col in numerical_cols:
        stats_by_col[col] = describe_column(data, col, show_bonus)
    
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    if show_bonus:
        stat_names.extend(['Variance', 'Skewness', 'Kurtosis'])
    
    col_widths = {}
    for col in numerical_cols:
        width = len(col) + 2
        width = max(width, 14)
        col_widths[col] = width

    label_width = 10

    print(f"{'':<{label_width}}", end='')
    for col in numerical_cols:
        print(f"{col:>{col_widths[col]}}", end='')
    print()
    
    for stat in stat_names:
        print(f"{stat:<{label_width}}", end='')
        for col in numerical_cols:
            if stats_by_col[col] is not None:
                value = stats_by_col[col].get(stat)
                if value is not None:
                    print(f"{value:>{col_widths[col]}.6f}", end='')
                else:
                    print(f"{'N/A':>{col_widths[col]}}", end='')
            else:
                print(f"{'N/A':>{col_widths[col]}}", end='')
        print()


def main():
    show_bonus = False
    
    if "--bonus" in sys.argv:
        show_bonus = True
        sys.argv.remove("--bonus")
    
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <dataset> [--bonus]")
        return

    header, data = read_csv(sys.argv[1])
    
    if data is None:
        return
    
    numerical_cols = get_numerical_columns(header)
    
    if not numerical_cols:
        print("Aucune colonne numérique trouvée dans le dataset")
        return
    
    print_table(data, numerical_cols, show_bonus)


if __name__ == "__main__":
    main()
