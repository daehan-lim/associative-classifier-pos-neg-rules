import csv
import numpy as np

records = []
with open('../../data/drugnames_test.csv', 'r') as file:
    for row in csv.reader(file):
        records.append(list(filter(None, row)))
drug_names = np.array(records).flatten().tolist()
result = ''
for drug_name in drug_names:
    drug_name = drug_name.replace("'", "''")
    result = f"{result}CASE WHEN MAX(CASE WHEN drugname = '{drug_name}' THEN 1 ELSE 0 END) = 1 THEN '{drug_name}' END as \"{drug_name[0:50]}\",\n"
    # result = f"{result}MAX(CASE WHEN drugname = '{drug_name}' THEN 1 ELSE 0 END) as \"{drug_name[0:60]}\",\n"
result
