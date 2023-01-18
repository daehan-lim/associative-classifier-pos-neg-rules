import csv
import timeit

import numpy as np
import pandas as pd

from src.classification import classification
from src.rule_gen import rule_generation
from src.rule_gen import rule_gen_with_ck

if __name__ == '__main__':
    # dataset = pd.read_csv('data/store_data.csv', header=None)  # To make sure the first row is not thought of as
    # the heading for i in range(0, dataset.shape[0]): transactions.append([str(dataset.values[i, j]) for j in range(
    # 0, 20)])

    records = []
    with open('../data/eicu.csv', 'r') as file:
        for row in csv.reader(file):
            records.append(list(filter(None, row)))

    PCR, NCR = rule_generation.classification_rule_generation(transactions=records,
                                                              min_support=0.05, min_conf=0.3, corr=0.05)

    itemset = frozenset(['2 ML  -  METOCLOPRAMIDE HCL 5 MG/ML IJ SOLN', '3 ML VIAL : INSULIN LISPRO (HUMAN) 100 UNIT/ML SC SOLN'])
    sorted_rules = sorted(PCR + NCR, key=lambda d: abs(d['confidence']), reverse=True)
    predicted_class = classification.classification(itemset, sorted_rules, 0.1)
    pr = np.expand_dims(np.array(PCR), axis=1)
    nr = np.expand_dims(np.array(NCR), axis=1)
    print(PCR)
    print(NCR)
    print(f"itemset: {itemset}")
    print(f"Predicted class: {predicted_class}")

    # print(rule_generation.classification_rule_generation(transactions=records,
    #                                                      min_support=0.05, min_conf=0.2, corr=0.3))

    # print(rule_gen_with_ck.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1))

    # print(timeit.timeit(lambda: rule_generation.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1), number=5))

    # print(timeit.timeit(lambda: rule_gen_with_ck.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1), number=5))
