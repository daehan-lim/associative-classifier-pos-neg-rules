from sklearn.metrics import classification_report, roc_auc_score
import csv
import numpy as np
from tabulate import tabulate
from classification import classification
from rule_gen import rule_generation
from daehan_mlutil import utilities


@utilities.timeit
def main():
    with open('../data/training_set_big_h.csv', 'r') as file:
        training_set = [list(filter(None, row)) for row in csv.reader(file)]

    with open('../data/test_set_big_h.csv', 'r') as file:
        test_set = [list(filter(None, row)) for row in csv.reader(file)]

    # avg_transaction_size = sum(len(transaction) for transaction in training_set) / len(training_set)
    # min_transaction_size = min(len(transaction) for transaction in training_set)
    # max_transaction_size = max(len(transaction) for transaction in training_set)

    min_support = 0.07
    min_conf = 0.07
    corr = 0.001
    print(f"supp = {min_support},  conf = {min_conf}, \n")

    rules, freq_itemsets = rule_generation.classification_rule_generation(transactions=training_set,
                                                           m_min_support=min_support, m_min_conf=min_conf)
    rules_0 = [rule for rule in rules if rule['consequent'] == '0']
    rules_1 = [rule for rule in rules if rule['consequent'] == '1']

    sorted_rules = sorted(rules, key=lambda d: d['confidence'], reverse=True)

    y_test, y_pred, not_classified = predict(test_set, sorted_rules)

    TP = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_test == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_test == 1))

    confusion_matrix = [
        ["1=died  0=alive", "Pred class = '1'", "Pred class = '0'", 'Total actual c'],
        ["Actual class = '1'", str(TP) + "\n(TP)", str(FN) + "\n(FN)",
         str(TP + FN) + "\n(Total actual c= '1')"],
        ["Actual class = '0'", str(FP) + "\n(FP)", str(TN) + "\n(TN)",
         str(FP + TN) + "\n(Total actual c= '0')"],
        ["Total pred c", str(TP + FP) + "\n(Total pred as '1')", str(FN + TN) + "\n(Total pred as '0')",
         str(len(test_set))],
    ]
    print(tabulate(confusion_matrix, headers='firstrow', tablefmt='fancy_grid'))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = 100 * np.sum(y_test == y_pred) / len(y_test)
    print(f"Pred as -1: {not_classified}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1 (harmonic mean): {round(F1, 3)}")
    print('roc auc: %.3f' % roc_auc_score(y_test, y_pred))
    print(f"Accuracy: {round(accuracy, 3)}%")
    print(f"Total Rules: {len(sorted_rules)}")
    print(f"Rules with class 0: {len(rules_0)}")
    print(f"Rules with class 1: {len(rules_1)}")
    print(f"Avg rule conf: {round(sum(rule['confidence'] for rule in sorted_rules) / len(sorted_rules), 3)}")
    print(f"Max rule conf: {round(sorted_rules[0]['confidence'], 3)}")
    print(f"Min rule conf: {round(sorted_rules[-1]['confidence'], 3)}\n")
    sorted_0 = [rule for rule in sorted_rules if rule['consequent'] == '0']
    sorted_1 = [rule for rule in sorted_rules if rule['consequent'] == '1']
    print(f"Avg conf for c0 rules: {round(sum(rule['confidence'] for rule in sorted_0) / len(sorted_0), 3)}")
    print(f"Max conf for c0 rules: {round(sorted_0[0]['confidence'], 3)}")
    print(f"Min conf for c0 rules: {round(sorted_0[-1]['confidence'], 3)}")
    print(f"Avg conf for c1 rules: {round(sum(rule['confidence'] for rule in sorted_1) / len(sorted_1), 3)}")
    print(f"Max conf for c1 rules: {round(sorted_1[0]['confidence'], 3)}")
    print(f"Min conf for c1 rules: {round(sorted_1[-1]['confidence'], 3)}")
    print(f"Max length of freq itemsets (k): {len(freq_itemsets) - 1}")
    print(f"Length of f1: {len(freq_itemsets[0])}")
    try:
        print(f"Length of f2: {len(freq_itemsets[1])}")
        print(f"Length of f3: {len(freq_itemsets[2])}")
        print(f"Length of f4: {len(freq_itemsets[3])}")
        print(f"Length of f5: {len(freq_itemsets[4])}")
        print(f"Length of last fk: {len(freq_itemsets[-2])}\n")
    except TypeError:
        pass
    print(classification_report(y_test, y_pred, zero_division=0))
    print()

    # pr = np.expand_dims(np.array(PCR), axis=1)
    # print(timeit.timeit(lambda: rule_generation.classification_rule_generation(), number=1))


@utilities.timeit
def predict(test_set, sorted_rules):
    y_test = []
    y_pred = []
    for transaction in test_set:
        y_test.append(int(transaction[-1]))
        object_o = frozenset([item for item in transaction[:-1]])
        y_pred.append(classification.classification(object_o, sorted_rules, 0.1))
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    not_classified = np.sum(y_pred == -1)
    y_pred[y_pred == -1] = 0
    return y_test, y_pred, not_classified


if __name__ == '__main__':
    main()
