import csv
import math

def read_csv(filepath):
    try:
        with open(filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            header = reader.fieldnames
            data = {col: [] for col in header}

            for row in reader:
                for col in header:
                    data[col].append(row[col])
            return header, data
    except FileNotFoundError:
        print(f"Erreur: le fichier {filepath} n'existe pas")
        return None, None

def get_numerical_columns(header):
    non_numerical = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    numerical_cols = []

    for col in header:
        if col not in non_numerical:
            numerical_cols.append(col)
    
    return numerical_cols

def get_data_by_column(data, col_name):
    values = []
    
    for val in data[col_name]:
        if val == '' or val is None:
            continue
        try:
            values.append(float(val))
        except ValueError:
            continue
    
    return values

def calculateMean(values):
    if len(values) == 0:
        return None
    total = 0
    for num in values:
        total += num
    return total / len(values)

def calculateStandardDeviation(values, mean):
    if len(values) == 0:
        return None
    total = 0
    for num in values:
        total += math.pow(num - mean, 2)
    return math.sqrt(total / len(values))