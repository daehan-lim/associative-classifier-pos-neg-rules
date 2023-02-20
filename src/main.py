from sklearn.metrics import classification_report
import csv
import numpy as np
from tabulate import tabulate
from util import util
import timeit

from classification import classification
from rule_gen import rule_generation

if __name__ == '__main__':

    with open('../data/training_set_norm.csv', 'r') as file:
        training_set = [list(filter(None, row)) for row in csv.reader(file)]

    with open('../data/test_set_norm.csv', 'r') as file:
        test_set = [list(filter(None, row)) for row in csv.reader(file)]

    # avg_transaction_size = sum(len(transaction) for transaction in training_set) / len(training_set)
    # min_transaction_size = min(len(transaction) for transaction in training_set)
    # max_transaction_size = max(len(transaction) for transaction in training_set)

    min_support = 0.03
    min_conf = 0.05
    corr = 0.001
    print(f"supp = {min_support},  conf = {min_conf}, ")

    rules = rule_generation.classification_rule_generation(
        transactions=training_set, classes=[frozenset(['0']), frozenset(['1'])], m_min_support=min_support,
        m_min_conf=min_conf)
    sorted_rules = sorted(rules, key=lambda d: d['confidence'], reverse=True)

    real_classes = []
    predicted_classes = []
    predicted_probs = []
    transactions_df = util.convert_trans_to_df(test_set)
    for transaction in test_set:
        real_classes.append(int(transaction[-1]))
        object_o = frozenset([item for item in transaction[:-1]])
        predicted_classes.append(classification.classification(object_o, sorted_rules, 0.1))
    y_true, y_pred = np.array(real_classes), np.array(predicted_classes)
    accuracy = 100 * np.sum(y_true == y_pred) / len(y_true)

    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    # Predicted a label of 1 (Alive), but the true label is 0.
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    # Predicted a label of 0 (Dead), but the true label is 1.
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    # Predicted as -1 when actual class = 1 (positive)
    NO_P = np.sum(np.logical_and(y_pred == -1, y_true == 1))
    # Predicted as -1 when actual class = 0
    NO_N = np.sum(np.logical_and(y_pred == -1, y_true == 0))

    confusion_matrix = [
        ["1=died  0=alive", "Pred class = '1'", "Pred class = '0'", "Pred c = '-1'", 'Total actual c'],
        ["Actual class = '1'", str(TP) + "\n(TP)", str(FN) + "\n(FN)",
         str(NO_P) + "\n(Pred as -1, actual = 1)", str(TP + FN + NO_P) + "\n(Total actual c= '1')"],
        ["Actual class = '0'", str(FP) + "\n(FP)", str(TN) + "\n(TN)",
         str(NO_N) + "\n(Pred as -1, actual = 0)", str(FP + TN + NO_N) + "\n(Total actual c= '0')"],
        ["Total pred c",
         str(TP + FP) + "\n(Total pred as '1')",
         str(FN + TN) + "\n(Total pred as '0')",
         str(NO_P + NO_N) + "\n(Total pred as '-1')", str(len(test_set))],
    ]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * recall * precision / (recall + precision)
    precision_w_unclass = TP / (TP + FP + NO_N)
    recall_w_unclass = TP / (TP + FN + NO_P)
    F1_w_unclass = 2 * recall_w_unclass * precision_w_unclass / (recall_w_unclass + precision_w_unclass)
    print(tabulate(confusion_matrix, headers='firstrow', tablefmt='fancy_grid'))
    # print(f"Precision considering -1 class: {round(precision_w_unclass, 3)}")
    # print(f"Recall considering -1 class: {round(recall_w_unclass, 3)}")
    # print(f"F1 (harmonic mean) considering -1 class: {round(F1_w_unclass, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1 (harmonic mean): {round(F1, 3)}")
    # print(f"F1 mean: {round(2 * F1 * F1_w_unclass / (F1 + F1_w_unclass), 3)}")
    print(f"Accuracy: {accuracy}%")
    print(f"Rules: {len(rules)}")
    print(f"Max length of freq itemsets (k): {len(rules[-1]['antecedent'])}")
    print(f"Avg rule conf: {round(sum(rule['confidence'] for rule in rules) / len(rules), 3)}")
    print(f"Max rule conf: {round(sorted_rules[0]['confidence'], 3)}")
    print(f"Min rule conf: {round(sorted_rules[-1]['confidence'], 3)}")
    print(classification_report(y_true, y_pred))
    # PR AUC is the average of precision scores calculated for each recall threshold.
    # print(f"PR AUC: {round(average_precision_score(y_true, predicted_probs), 3)}")
    #
    # fprs, tprs, thresholds = roc_curve(y_true, predicted_probs)
    #
    # plt.plot(fprs, tprs, label='ROC curve (area = %.2f)' % metrics.auc(fprs, tprs))
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    # plt.title('ROC curve')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.grid()
    # plt.legend()
    # plt.show()

    # pr = np.expand_dims(np.array(PCR), axis=1)
    # print(timeit.timeit(lambda: rule_generation.classification_rule_generation(), number=1))
