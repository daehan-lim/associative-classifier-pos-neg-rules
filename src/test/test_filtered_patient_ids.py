import csv
import numpy as np

records = []
with open('../../data/hospitals_w_more_t_800_p.csv', 'r') as file:
    for row in csv.reader(file):
        records.append(list(filter(None, row)))
hospital_ids = np.array(records).flatten().tolist()
result = ''
for hospital_id in hospital_ids:
    result += f"""
    UNION
    SELECT distinct * from (
        SELECT *
        FROM (
            SELECT distinct patient.patientunitstayid
            FROM patient
            JOIN hospital h on patient.hospitalid = h.hospitalid
            JOIN medication m on patient.patientunitstayid = m.patientunitstayid
            WHERE h.hospitalid = {hospital_id}
              AND m.drugname is not null
        ) as distinct_patients
        ORDER BY random()
        LIMIT 560
    ) as distinct_random
    """
result
