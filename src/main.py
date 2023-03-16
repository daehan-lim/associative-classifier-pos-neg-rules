import random
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import csv
import numpy as np
from tabulate import tabulate
from classification import classification
from rule_gen import rule_generation
from daehan_mlutil import utilities
from util import util


@utilities.timeit
def main():
    with open('../data/dataset.csv', 'r') as file:
        dataset = [list(filter(None, row)) for row in csv.reader(file)]

    min_support = 0.06
    transactions_df = util.convert_trans_to_df(dataset)
    auc_sum = 0
    f1_sum = 0
    for seed in range(10):
        print(f"\n\nseed: {seed}")
        print(f"min_supp: {min_support}")
        random.seed(seed)
        transactions_0 = pd.DataFrame(
            transactions_df[transactions_df['0']].reset_index(drop=True))
        transactions_1 = pd.DataFrame(
            transactions_df[transactions_df['1']].reset_index(drop=True))

        indices = list(range(0, len(transactions_0)))
        random.shuffle(indices)
        test_set_0 = transactions_0.iloc[indices[:417], :].reset_index(drop=True)
        training_set_0 = transactions_0.iloc[indices[417:], :].reset_index(drop=True)

        indices = list(range(0, len(transactions_1)))
        random.shuffle(indices)
        test_set_1 = transactions_1.iloc[indices[:43], :].reset_index(drop=True)
        training_set_1 = transactions_1.iloc[indices[43:], :].reset_index(drop=True)

        training_set = pd.concat([training_set_0, training_set_1])
        test_set = pd.concat([test_set_0, test_set_1])

        rules, freq_itemsets = rule_generation.classification_rule_generation(transactions=training_set,
                                                                              m_min_support=min_support)
        rules_0 = [rule for rule in rules if rule['consequent'] == '0']
        rules_1 = [rule for rule in rules if rule['consequent'] == '1']

        sorted_rules = sorted(rules, key=lambda d: d['confidence'], reverse=True)

        y_test, y_pred, scores = predict(test_set, training_set, sorted_rules)

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
        f1_sum += F1
        roc_auc = roc_auc_score(y_test, scores)
        auc_sum += roc_auc
        accuracy = 100 * np.sum(y_test == y_pred) / len(y_test)
        # print(f"Pred as -1: {not_classified}")
        print(f"Precision: {round(precision, 3)}")
        print(f"Recall: {round(recall, 3)}")
        print(f"F1: {round(F1, 6)}")
        print(f"roc auc: {roc_auc}")
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

    print("\n\nAvg")
    print(f"Roc auc (class 1): {auc_sum / 10}")
    print(f"f1: {f1_sum / 10}")


@utilities.timeit
def predict(test_set, training_set, sorted_rules):
    training_transactions_1 = training_set[training_set['1']].drop(['1', '0'], axis=1).apply(
        lambda row: frozenset(row.index[row]), axis=1).tolist()
    scores_training = [classification.predict_proba(object_o, sorted_rules) for object_o in training_transactions_1]
    mean = np.mean(scores_training)

    test_transactions = test_set.drop(['1', '0'], axis=1).apply(lambda row: frozenset(row.index[row]), axis=1).tolist()
    scores_test = [classification.predict_proba(object_o, sorted_rules) for object_o in test_transactions]
    scores_test = np.array(scores_test)
    y_test = test_set.apply(lambda row: 0 if row['0'] else 1, axis=1).tolist()
    y_test = np.array(y_test)
    y_pred = np.zeros(len(scores_test), dtype=int)
    y_pred[scores_test >= mean] = 1
    return y_test, y_pred, scores_test


if __name__ == '__main__':
    main()
