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

def get_data_by_column_with_none(data, col_name):
    values = []
    
    for val in data[col_name]:
        if val == '' or val is None:
            values.append(None)
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

def findMin(values):
    if len(values) == 0:
        return None
    
    min_val = values[0]
    for num in values:
        if num < min_val:
            min_val = num
    
    return min_val


def findMax(values):
    if len(values) == 0:
        return None
    
    max_val = values[0]
    for num in values:
        if num > max_val:
            max_val = num
    
    return max_val


def calculatePercentile(values, percentile):
    if len(values) == 0:
        return None
    
    sorted_values = sorted(values)
    
    index = (percentile / 100.0) * (len(sorted_values) - 1)
    
    lower_index = int(index)
    upper_index = lower_index + 1
    
    if upper_index >= len(sorted_values):
        return sorted_values[lower_index]
    
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = index - lower_index
    
    percentile_value = lower_value + weight * (upper_value - lower_value)
    
    return percentile_value
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
