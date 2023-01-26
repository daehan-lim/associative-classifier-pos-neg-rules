import csv
import numpy as np

records = []
with open('../../data/eicu_drugnames.csv', 'r') as file:
    for row in csv.reader(file):
        records.append(list(filter(None, row)))
drug_names = np.array(records).flatten().tolist()
result = ''
for drug_name in drug_names:
    drug_name = drug_name.replace("'", "''")
    result = f"{result}MAX(CASE WHEN drugname = '{drug_name}' THEN 1 ELSE 0 END) as \"{drug_name[0:60]}\",\n"
result