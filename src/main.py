import csv
import numpy as np
import pandas as pd
from src.classification import classification
from src.rule_gen import rule_generation

if __name__ == '__main__':
    # dataset = pd.read_csv('data/store_data.csv', header=None)  # To make sure the first row is not thought of as
    # the heading for i in range(0, dataset.shape[0]): transactions.append([str(dataset.values[i, j]) for j in range(
    # 0, 20)])

    with open('../data/eicu_all_attrb_header.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        header.remove('mortality')
        c1 = [frozenset([item]) for item in header]
        records = [list(filter(None, row)) for row in reader]

        # df = pd.read_csv('../data/eicu_all_attrb_header.csv', keep_default_na=False)
        # header2 = df.columns.tolist()
        # header2.remove('mortality')
        # c1_2 = [frozenset([item]) for item in header2]
        # # noinspection PyTypeChecker
        # records2 = [list(filter(None, row)) for row in df.values.tolist()]

    object_o = frozenset(['ACETAMINOPHEN 325 MG PO TABS',
                          ])

    PCR, NCR = rule_generation.classification_rule_generation(
        transactions=records, c1=c1, classes=[frozenset(['Alive']), frozenset(['Expired'])], min_support=0.01,
        min_conf=0.5, corr=0.05)
    sorted_rules = sorted(PCR + NCR, key=lambda d: abs(d['confidence']), reverse=True)
    predicted_class = classification.classification(object_o, sorted_rules, 0.1)

    pr = np.expand_dims(np.array(PCR), axis=1)
    nr = np.expand_dims(np.array(NCR), axis=1)
    print(PCR)
    print(NCR)
    print(f"Object to classify: {object_o}")
    print(f"Predicted class: {predicted_class}")

    # print(rule_generation.classification_rule_generation(transactions=records,
    #                                                      min_support=0.05, min_conf=0.2, corr=0.3))

    # print(rule_gen_with_ck.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1))

    # print(timeit.timeit(lambda: rule_generation.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1), number=5))

    # print(timeit.timeit(lambda: rule_gen_with_ck.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1), number=5))
