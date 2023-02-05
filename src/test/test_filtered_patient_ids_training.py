import csv
import numpy as np

records = []
with open('../../data/hospitals_w_more_t_800_p.csv', 'r') as file:
    for row in csv.reader(file):
        records.append(list(filter(None, row)))
hospital_ids = np.array(records).flatten().tolist()
result = ''
for hospital_id in hospital_ids:
    result += \
    f"""
    UNION
    SELECT distinct *
    FROM
        (SELECT *
         FROM results.filtered_patient_ids
         WHERE  filtered_patient_ids.patientunitstayid in
                (SELECT distinct patient.patientunitstayid
                 FROM patient
                 JOIN hospital h on patient.hospitalid = h.hospitalid
                 WHERE h.hospitalid = {hospital_id})
         ORDER BY filtered_patient_ids.patientunitstayid
         LIMIT 400) as query
    """
result
