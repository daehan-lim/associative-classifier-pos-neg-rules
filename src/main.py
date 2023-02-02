import csv
import numpy as np
from classification import classification
from rule_gen import rule_generation

if __name__ == '__main__':

    with open('../data/training_dataset.csv', 'r') as file:
        training_set = [list(filter(None, row)) for row in csv.reader(file)]

    # avg_transaction_size = sum(len(transaction) for transaction in training_set) / len(training_set)

    with open('../data/test_dataset.csv', 'r') as file:
        test_set = [list(filter(None, row)) for row in csv.reader(file)]

    min_support = 0.03
    min_conf = 0.15
    corr = 0.07
    rules = rule_generation.classification_rule_generation(
        transactions=training_set, m_classes=[frozenset(['1']), frozenset(['0'])], m_min_support=min_support,
        m_min_conf=min_conf)
    sorted_rules = sorted(rules, key=lambda d: abs(d['confidence']), reverse=True)

    real_classes = []
    predicted_classes = []
    for transaction in test_set:
        real_classes.append(int(transaction[-1]))
        object_o = frozenset([item for item in transaction[:-1]])
        predicted_classes.append(classification.classification(object_o, sorted_rules))
    y_true, y_pred = np.array(real_classes), np.array(predicted_classes)
    accuracy = 100 * np.sum(y_true == y_pred) / len(y_true)

    print(f"min_support = {min_support},  min_conf = {min_conf},  corr = {corr}")
    print(f"Out of {np.count_nonzero(y_true == 0)} '0' classes in test set")
    print(f"# of classes predicted as '0': {np.count_nonzero(y_pred == 0)}")
    print(f"# of misclassifications: {np.count_nonzero(y_pred == -1)}")
    print(f"Rules: {len(rules)}")
    print(f"Accuracy: {accuracy}%")

    # pr = np.expand_dims(np.array(PCR), axis=1)
    # print(timeit.timeit(lambda: rule_generation.classification_rule_generation(
    #     transactions=records, min_support=0.005, min_conf=0.2, corr=1), number=5))
