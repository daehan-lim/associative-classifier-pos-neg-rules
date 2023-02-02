import csv
import numpy as np
from src.classification import classification
from src.rule_gen import rule_generation

if __name__ == '__main__':

    with open('../data/training_dataset.csv', 'r') as file:
        training_set = [list(filter(None, row)) for row in csv.reader(file)]

    avg_transaction_size = sum(len(transaction) for transaction in training_set) / len(training_set)

    with open('../data/test_dataset.csv', 'r') as file:
        test_set = [list(filter(None, row)) for row in csv.reader(file)]

    min_support = 0.03
    min_conf = 0.15
    corr = 0.07
    PCR, NCR = rule_generation.classification_rule_generation(
        transactions=training_set, classes=[frozenset(['1']), frozenset(['0'])], min_support=min_support,
        min_conf=min_conf, corr=corr)
    sorted_rules = sorted(PCR + NCR, key=lambda d: abs(d['confidence']), reverse=True)

    real_classes = []
    predicted_classes = []
    for transaction in test_set:
        real_classes.append(int(transaction[-1]))
        object_o = frozenset([item for item in transaction[:-1]])
        predicted_classes.append(classification.classification(object_o, sorted_rules))
    y_true, y_pred = np.array(real_classes), np.array(predicted_classes)
    accuracy = 100 * np.sum(y_true == y_pred) / len(y_true)

    print(f"min_support = {min_support},  min_conf = {min_conf},  corr = {corr}")
    print(f"# of 0 in y_true: {np.count_nonzero(y_true == 0)}")
    print(f"# of 0 in y_pred: {np.count_nonzero(y_pred == 0)}")
    print(f"# of -1 in y_pred: {np.count_nonzero(y_pred == -1)}")
    print(f"Positive rules: {len(PCR)}")
    print(f"Negative rules: {len(NCR)}")
    print(f"Accuracy: {accuracy}%")

    # pr = np.expand_dims(np.array(PCR), axis=1)
    # print(timeit.timeit(lambda: rule_generation.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1), number=5))
