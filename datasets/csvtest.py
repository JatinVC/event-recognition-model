import csv
from typing import Dict
import numpy as np

def load_origin_data(file_name: str) -> Dict:
    data = {
        'x': np.array([]),
        'y': [],
        'p': [],
        't': []
    }

    csvReader = csv.reader(open(file_name), delimiter=',')
    for line in csvReader: 
        print(line)
        np.append(data['x'], line[0])
        np.append(data['y'], line[1])
        np.append(data['p'], line[2])
        np.append(data['t'], line[3])

    return data

print(load_origin_data('data.txt'))