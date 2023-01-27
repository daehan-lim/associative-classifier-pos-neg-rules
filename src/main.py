import csv
import numpy as np
import pandas as pd
from src.classification import classification
from src.rule_gen import rule_generation

if __name__ == '__main__':

    with open('../data/training_dataset.csv', 'r') as file:
        training_set = [list(filter(None, row)) for row in csv.reader(file)]

    with open('../data/test_dataset.csv', 'r') as file:
        test_set = [list(filter(None, row)) for row in csv.reader(file)]
    first_transaction = test_set[0].copy()
    first_transaction.pop()
    first_object = frozenset([item for item in first_transaction])

    PCR, NCR = rule_generation.classification_rule_generation(
        transactions=training_set, classes=[frozenset(['1']), frozenset(['0'])], min_support=0.05,
        min_conf=0.05, corr=0.03)
    sorted_rules = sorted(PCR + NCR, key=lambda d: abs(d['confidence']), reverse=True)

    real_classes = []
    predicted_classes = []
    for transaction in test_set:
        real_classes.append(transaction[-1])
        object_o = frozenset([item for item in transaction[:-1]])
        predicted_classes.append(classification.classification(object_o, sorted_rules, 0.1))

    pr = np.expand_dims(np.array(PCR), axis=1)
    nr = np.expand_dims(np.array(NCR), axis=1)
    print(pr)
    print(NCR)
    print(f"Object to classify: {first_object}")
    print(f"Predicted class: {predicted_classes[0]}")

    # print(rule_generation.classification_rule_generation(transactions=records,
    #                                                      min_support=0.05, min_conf=0.2, corr=0.3))

    # print(rule_gen_with_ck.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1))

    # print(timeit.timeit(lambda: rule_generation.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1), number=5))

    # print(timeit.timeit(lambda: rule_gen_with_ck.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1), number=5))
