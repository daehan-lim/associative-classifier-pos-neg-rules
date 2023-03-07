import csv
import numpy as np
import copy
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import random
from mlxtend.frequent_patterns import apriori
from sklearn.metrics import roc_auc_score, classification_report
from tabulate import tabulate
import time

start_time = time.time()
with open('../data/dataset.csv', 'r') as file:
    data_set = [list(filter(None, row)) for row in csv.reader(file)]

te = TransactionEncoder()
te_ary = te.fit_transform(data_set)
m_transactions = pd.DataFrame(te_ary, columns=te.columns_)

transactions_0 = pd.DataFrame(
    m_transactions[m_transactions['0']].reset_index(drop=True).drop(['1', '0'], axis=1))
transactions_1 = pd.DataFrame(
    m_transactions[m_transactions['1']].reset_index(drop=True).drop(['1', '0'], axis=1))

# seeds [0, 10, 35, 42, 123, 456, 789, 101112, 131415, 161718]
auc_sum = 0
f1_sum = 0
for seed in range(10):
    print(f"\n\nseed: {seed}")
    random.seed(seed)
    indices = list(range(0, len(transactions_0)))
    random.shuffle(indices)
    transactions_te_0 = transactions_0.iloc[indices[:417], :]
    # transactions_tr_00 = transactions_0.iloc[indices[417:810],:]
    transactions_tr_0 = transactions_0.iloc[indices[417:], :]

    indices = list(range(0, len(transactions_1)))
    random.shuffle(indices)
    transactions_te_1 = transactions_1.iloc[indices[:43], :]
    transactions_tr_1 = transactions_1.iloc[indices[43:], :]

    tr_0_ary = transactions_tr_0.values.astype('int')
    tr_1_ary = transactions_tr_1.values.astype('int')

    # tr_00_ary=(transactions_tr_00.values).astype('int')

    transactions_tr = pd.concat([transactions_tr_0, transactions_tr_1])

    frequent_items = apriori(transactions_tr, min_support=0.1)

    attributes_count = transactions_tr.shape[1]  # number of attributes
    freq_itemsets_count = len(frequent_items)  # number of frequent items
    freq_itemsets = frequent_items['itemsets']
    freq_itemsets_matrix = [list(x) for x in freq_itemsets]
    attributes_contained_in_freq_items = np.zeros((attributes_count, freq_itemsets_count))
    for i in range(len(frequent_items)):
        attributes_contained_in_freq_items[freq_itemsets_matrix[i], i] = 1

    # --------------------------------------------------------------------------------------
    # # lift
    # print('lift')
    # cp_attr_contained_freq_items = copy.deepcopy(attributes_contained_in_freq_items)
    # freq_count_per_trans_0 = np.matmul(tr_0_ary, attributes_contained_in_freq_items)
    # freq_count_per_trans_1 = np.matmul(tr_1_ary, attributes_contained_in_freq_items)
    # # Each element in the matrix indicates the count of how many times the corresponding
    # # frequent itemset occurred in the transactions of the corresponding class.
    #
    # item_len = np.sum(attributes_contained_in_freq_items, axis=0)  # item length
    # cls0 = np.zeros(freq_itemsets_count)  # frequent_items['support']
    # cls1 = np.zeros(freq_itemsets_count)
    # p = np.zeros(freq_itemsets_count)
    # conf = np.zeros(freq_itemsets_count)
    # c_per_rule = np.zeros(freq_itemsets_count, dtype=int)
    # cnt = 0
    # for i in range(freq_count_per_trans_0.shape[1]):
    #     cls0[i] = (freq_count_per_trans_0[:, i] >= item_len[i]).sum() / freq_count_per_trans_0.shape[0]
    #     cls1[i] = (freq_count_per_trans_1[:, i] >= item_len[i]).sum() / freq_count_per_trans_1.shape[0]
    #     p[i] = ((freq_count_per_trans_0[:, i] >= item_len[i]).sum() + (
    #             freq_count_per_trans_1[:, i] >= item_len[i]).sum()) / (
    #                    freq_count_per_trans_0.shape[0] + freq_count_per_trans_1.shape[0])
    #
    #     '''
    #     p: proportion of transactions in which the corresponding frequent itemset appears
    #     it is calculated as the sum of the number of transactions containing the itemset
    #     in both classes (0 and 1) divided by the total number of transactions.
    #
    #     cls0[i]/p[i] is the ratio of the number of transactions in which the itemset appears in class 0
    #     to the total number of transactions in which the itemset appears, regardless of the class.
    #     The confidence score indicates the likelihood that the rule's consequent will be present in transactions that
    #     contain its antecedent.
    #
    #     The line elif cls0[i]/p[i]<1 and cls1[i]/p[i]>1: is checking whether the confidence of the association rule
    #     in question is greater for class 1 than class 0, and whether that confidence is greater than 1.
    #     If the confidence of the rule is less for class 0 than class 1, but the confidence for class 1 is greater than 1,
    #     then the rule is said to have a "class 1" association, meaning that it is more strongly associated with class 1.
    #     In this case, indc[i] is set to 1, indicating that the rule has a class 1 association,
    #     and conf[i] is set to the ratio of the support of the rule in class 1 to the overall support of the rule
    #     in both classes.
    #     '''
    #
    #     if cls0[i] / p[i] > 1:
    #         c_per_rule[i] = 0
    #         conf[i] = cls0[i] / p[i]
    #         cnt = cnt + 1
    #     elif cls0[i] / p[i] < 1 and cls1[i] / p[i] > 1:
    #         c_per_rule[i] = 1
    #         conf[i] = cls1[i] / p[i]
    #         cnt = cnt + 1
    #     else:
    #         conf[i] = 0
    #         cp_attr_contained_freq_items[:, i] = 0
    #         c_per_rule[i] = -1

    # 1-ccs/cls
    print('1-ccs/cls')
    cp_attr_contained_freq_items = copy.deepcopy(attributes_contained_in_freq_items)
    freq_count_per_trans_0 = np.matmul(tr_0_ary, attributes_contained_in_freq_items)  # x_00
    freq_count_per_trans_1 = np.matmul(tr_1_ary, attributes_contained_in_freq_items)  # x_01
    # Each element in the matrix indicates the count of how many times the corresponding
    # frequent itemset occurred in the transactions of the corresponding class.

    item_len = np.sum(attributes_contained_in_freq_items, axis=0)  # item length
    cls0 = np.zeros(freq_itemsets_count)  # frequent_items['support']
    cls1 = np.zeros(freq_itemsets_count)
    conf = np.zeros(freq_itemsets_count)
    c_per_rule = np.zeros(freq_itemsets_count, dtype=int)
    cnt = 0
    for i in range(freq_count_per_trans_0.shape[1]):
        cls0[i] = (freq_count_per_trans_0[:, i] >= item_len[i]).sum() / freq_count_per_trans_0.shape[0]
        cls1[i] = (freq_count_per_trans_1[:, i] >= item_len[i]).sum() / freq_count_per_trans_1.shape[0]
        if cls1[i] / cls0[i] < 1:
            c_per_rule[i] = 0
            conf[i] = 1 - cls1[i] / cls0[i]
            cnt = cnt + 1
        elif cls1[i] / cls0[i] > 1:
            c_per_rule[i] = 1
            conf[i] = 1 - cls0[i] / cls1[i]
            cnt = cnt + 1
        else:
            conf[i] = 0
            cp_attr_contained_freq_items[:, i] = 0
            c_per_rule[i] = -1


    # --------------------------------------------------------------------------------------
    # # first rule; roc auc
    # print('first rule')
    # print(f"freq items: {len(frequent_items)}")
    # print(f"Rules for class 0: {(c_per_rule == 0).sum()}; Rules for class 1: {(c_per_rule == 1).sum()}")
    # print(f"Avg conf for class 0: {conf[c_per_rule == 0].mean()}; Avg conf for class 1: {conf[c_per_rule == 1].mean()}")
    #
    # te_0_ary = transactions_te_0.values.astype('int')
    # te_1_ary = transactions_te_1.values.astype('int')
    #
    # # first rule used
    # y_0 = np.matmul(te_0_ary, cp_attr_contained_freq_items)
    # y_1 = np.matmul(te_1_ary, cp_attr_contained_freq_items)
    # # Each element in the matrix indicates the count of how many times the corresponding
    # # frequent itemset occurred in the transactions of the corresponding class.
    #
    # pred = np.zeros((y_0.shape[0] + y_1.shape[0], 2))
    # no_rules_x_c0 = 0
    # no_rules_x_c1 = 0
    # '''
    # no_rules_x_c0 and no_rules_x_c1 are variables that keep track of the number of cases where no rule applies for each class.
    # In the code, they are initialized to 0 before the for-loops, and then updated within the loops based on
    # whether or not a rule applies to a given transaction. If no rule applies, the corresponding count is incremented.
    # '''
    # for i in range(y_0.shape[0]):
    #     maxp = -1
    #     '''
    #      maxp is a variable that stores the maximum confidence value for a rule that satisfies certain conditions.
    #      In the first for loop, maxp is the maximum confidence value among rules that belong to class 0 and contain
    #      all the items in the current transaction. In the second for loop, maxp is the maximum confidence value among
    #      rules that belong to class 1 and contain all the items in the current transaction.
    #      If no rule satisfies the conditions, maxp remains -1.
    #     '''
    #     for j in range(y_0.shape[1]):
    #         if c_per_rule[j] == 0 and y_0[i, j] >= item_len[j]:
    #             maxp = max(maxp, conf[j])
    #     if maxp == -1:
    #         no_rules_x_c0 = no_rules_x_c0 + 1
    #     else:
    #         pred[i, 0] = maxp
    #
    #     maxp = -1
    #     for j in range(y_0.shape[1]):
    #         if c_per_rule[j] == 1 and y_0[i, j] >= item_len[j]:
    #             maxp = max(maxp, conf[j])
    #
    #     if maxp == -1:
    #         no_rules_x_c1 = no_rules_x_c1 + 1
    #     else:
    #         pred[i, 1] = maxp
    # print(f"using class 0: {y_0.shape[0], no_rules_x_c0, no_rules_x_c1}")
    #
    # no_rules_x_c0 = 0
    # no_rules_x_c1 = 0
    # # keep track of the number of cases where no rule applies for each class.
    #
    # for i in range(y_1.shape[0]):
    #     maxp = -1
    #     for j in range(y_1.shape[1]):
    #         if c_per_rule[j] == 0 and y_1[i, j] >= item_len[j]:
    #             maxp = max(maxp, conf[j])
    #     if maxp == -1:
    #         no_rules_x_c0 = no_rules_x_c0 + 1
    #     else:
    #         pred[i + y_0.shape[0], 0] = maxp
    #
    #     maxp = -1
    #     for j in range(y_1.shape[1]):
    #         if c_per_rule[j] == 1 and y_1[i, j] >= item_len[j]:
    #             maxp = max(maxp, conf[j])
    #
    #     if maxp == -1:
    #         no_rules_x_c1 = no_rules_x_c1 + 1
    #     else:
    #         pred[i + y_0.shape[0], 1] = maxp
    #
    # print(f"using class 1: {y_1.shape[0], no_rules_x_c0, no_rules_x_c1}")
    #
    # y = np.concatenate((np.zeros(te_0_ary.shape[0]), np.ones(te_1_ary.shape[0])), axis=0)
    # auc1 = roc_auc_score(y, pred[:, 1])
    # auc_sum += auc1
    # print(f"\nauc for class 1: {auc1}")
    # y = np.concatenate((np.zeros(te_0_ary.shape[0]), np.ones(te_1_ary.shape[0])), axis=0)
    # auc0 = roc_auc_score(y, -pred[:, 0])
    # print(f"auc for class 0: {auc0}")
    #
    #
    #
    #
    # # f1, first rule used for training data
    # y_00 = np.matmul(tr_0_ary, cp_attr_contained_freq_items)
    # y_01 = np.matmul(tr_1_ary, cp_attr_contained_freq_items)
    # # Each element in the matrix indicates the count of how many times the corresponding
    # # frequent itemset occurred in the transactions of the corresponding class.
    #
    # pred0 = np.zeros(y_00.shape[0] + y_01.shape[0])
    # no_rules_x_c0 = 0
    # no_rules_x_c1 = 0
    # for i in range(y_00.shape[0]):
    #     maxp = -1
    #     for j in range(y_00.shape[1]):
    #         if c_per_rule[j] == 1 and y_00[i, j] >= item_len[j]:
    #             maxp = max(maxp, conf[j])
    #
    #     if maxp == -1:
    #         no_rules_x_c1 = no_rules_x_c1 + 1
    #     else:
    #         pred0[i] = maxp
    #
    # no_rules_x_c0 = 0
    # no_rules_x_c1 = 0
    # for i in range(y_01.shape[0]):
    #
    #     maxp = -1
    #     for j in range(y_01.shape[1]):
    #         if c_per_rule[j] == 1 and y_01[i, j] >= item_len[j]:
    #             maxp = max(maxp, conf[j])
    #
    #     if maxp == -1:
    #         no_rules_x_c1 = no_rules_x_c1 + 1
    #     else:
    #         pred0[i + y_00.shape[0]] = maxp
    #
    # y0 = np.concatenate((np.zeros(tr_0_ary.shape[0]), np.ones(tr_1_ary.shape[0])), axis=0)
    #
    # z_1 = pred0[y0 == 1]
    # m = np.mean(z_1)
    # s = np.std(z_1)
    # th = m
    #
    # pred_y = np.zeros(pred.shape[0], dtype=int)
    # for i in range(pred.shape[0]):
    #     if pred[i, 1] >= th:
    #         pred_y[i] = 1
    #
    # # print(f1_score(y, pred_y))
    # TP = 0
    # FP = 0
    # FN = 0
    # TN = 0
    # for i in range(pred.shape[0]):
    #     if y[i] == 1 and pred_y[i] == 1:
    #         TP = TP + 1
    #     elif y[i] == 1 and pred_y[i] == 0:
    #         FN = FN + 1
    #     elif y[i] == 0 and pred_y[i] == 1:
    #         FP = FP + 1
    #     else:
    #         TN = TN + 1
    #
    # f1 = (2 * TP) / (2 * TP + FP + FN)
    # f1_sum += f1
    # print('pre:', TP / (TP + FP), 'rec:', TP / (TP + FN))
    # print('f1:', f1)
    #
    # confusion_matrix = [
    #     ["1=died  0=alive", "Pred class = '1'", "Pred class = '0'", 'Total actual c'],
    #     ["Actual class = '1'", str(TP) + "\n(TP)", str(FN) + "\n(FN)",
    #      str(TP + FN) + "\n(Total actual c= '1')"],
    #     ["Actual class = '0'", str(FP) + "\n(FP)", str(TN) + "\n(TN)",
    #      str(FP + TN) + "\n(Total actual c= '0')"],
    #     ["Total pred c", str(TP + FP) + "\n(Total pred as '1')", str(FN + TN) + "\n(Total pred as '0')",
    #      str(pd.concat([transactions_te_0, transactions_te_1]).shape[0])],
    # ]
    # print(tabulate(confusion_matrix, headers='firstrow', tablefmt='fancy_grid'))
    # print(classification_report(y, pred_y, zero_division=0))




    # --------------------------------------------------------------------------------------
    # all rules; roc auc
    print('all rules')
    print(f"freq items: {len(frequent_items)}")
    print(f"Rules for class 0: {(c_per_rule == 0).sum()} Rules for class 1: {(c_per_rule == 1).sum()}")
    print(f"Avg conf for class 0: {conf[c_per_rule == 0].mean()}, Avg conf for class 1: {conf[c_per_rule == 1].mean()}")

    te_0_ary = transactions_te_0.values.astype('int')
    te_1_ary = transactions_te_1.values.astype('int')

    # all matched rules  used
    y_0 = np.matmul(te_0_ary, cp_attr_contained_freq_items)
    y_1 = np.matmul(te_1_ary, cp_attr_contained_freq_items)
    # Each element in the matrix indicates the count of how many times the corresponding
    # frequent itemset occurred in the transactions of the corresponding class.

    pred = np.zeros((y_0.shape[0] + y_1.shape[0], 2))
    no_rules_x_c0 = 0
    no_rules_x_c1 = 0

    for i in range(y_0.shape[0]):
        cnt = 0
        for j in range(y_0.shape[1]):
            if c_per_rule[j] == 0 and y_0[i, j] >= item_len[j]:
                pred[i, 0] = pred[i, 0] + conf[j]
                cnt = cnt + 1
        if cnt == 0:
            no_rules_x_c0 = no_rules_x_c0 + 1
        else:
            pred[i, 0] = pred[i, 0] / cnt

        cnt = 0
        for j in range(y_0.shape[1]):
            if c_per_rule[j] == 1 and y_0[i, j] >= item_len[j]:
                pred[i, 1] = pred[i, 1] + conf[j]
                cnt = cnt + 1

        if cnt == 0:
            no_rules_x_c1 = no_rules_x_c1 + 1
        else:
            pred[i, 1] = pred[i, 1] / cnt
    print(f"using class 0: {y_0.shape[0], no_rules_x_c0, no_rules_x_c1}")

    no_rules_x_c0 = 0
    no_rules_x_c1 = 0
    for i in range(y_1.shape[0]):
        cnt = 0
        x = 0
        for j in range(y_1.shape[1]):
            if c_per_rule[j] == 0 and y_1[i, j] >= item_len[j]:
                x = x + conf[j]
                cnt = cnt + 1
        if cnt == 0:
            no_rules_x_c0 = no_rules_x_c0 + 1
        else:
            pred[i + y_0.shape[0], 0] = x / cnt

        cnt = 0
        x = 0
        for j in range(y_1.shape[1]):
            if c_per_rule[j] == 1 and y_1[i, j] >= item_len[j]:
                x = x + conf[j]
                cnt = cnt + 1

        if cnt == 0:
            no_rules_x_c1 = no_rules_x_c1 + 1
        else:
            pred[i + y_0.shape[0], 1] = x / cnt
    print(f"using class 1: {y_1.shape[0], no_rules_x_c0, no_rules_x_c1}")

    y = np.concatenate((np.zeros(te_0_ary.shape[0]), np.ones(te_1_ary.shape[0])), axis=0)
    auc1 = roc_auc_score(y, pred[:, 1])
    auc_sum += auc1
    print(f"\nauc for class 1: {auc1}")
    y = np.concatenate((np.zeros(te_0_ary.shape[0]), np.ones(te_1_ary.shape[0])), axis=0)
    auc0 = roc_auc_score(y, -pred[:, 0])
    print(f"auc for class 0: {auc0}")



    # f1, all rules used for training data
    y_00 = np.matmul(tr_0_ary, cp_attr_contained_freq_items)
    y_01 = np.matmul(tr_1_ary, cp_attr_contained_freq_items)
    # Each element in the matrix indicates the count of how many times the corresponding
    # frequent itemset occurred in the transactions of the corresponding class.

    pred0 = np.zeros(y_00.shape[0] + y_01.shape[0])
    no_rules_x_c0 = 0
    no_rules_x_c1 = 0
    for i in range(y_00.shape[0]):
        cnt = 0
        for j in range(y_00.shape[1]):
            if c_per_rule[j] == 1 and y_00[i, j] >= item_len[j]:
                pred0[i] = pred0[i] + conf[j]
                cnt = cnt + 1

        if cnt == 0:
            no_rules_x_c1 = no_rules_x_c1 + 1
        else:
            pred0[i] = pred0[i] / cnt

    no_rules_x_c0 = 0
    no_rules_x_c1 = 0
    for i in range(y_01.shape[0]):

        cnt = 0
        for j in range(y_01.shape[1]):
            if c_per_rule[j] == 1 and y_01[i, j] >= item_len[j]:
                pred0[i + y_00.shape[0]] = pred0[i + y_00.shape[0]] + conf[j]
                cnt = cnt + 1

        if (cnt == 0):
            no_rules_x_c1 = no_rules_x_c1 + 1
        else:
            pred0[i + y_00.shape[0]] = pred0[i + y_00.shape[0]] / cnt

    y0 = np.concatenate((np.zeros(tr_0_ary.shape[0]), np.ones(tr_1_ary.shape[0])), axis=0)

    z_1 = pred0[y0 == 1]
    m = np.mean(z_1)
    s = np.std(z_1)
    th = m

    pred_y = np.zeros(pred.shape[0], dtype=int)
    for i in range(pred.shape[0]):
        if pred[i, 1] >= th:
            pred_y[i] = 1

    # print(f1_score(y, pred_y))
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(pred.shape[0]):
        if y[i] == 1 and pred_y[i] == 1:
            TP = TP + 1
        elif y[i] == 1 and pred_y[i] == 0:
            FN = FN + 1
        elif y[i] == 0 and pred_y[i] == 1:
            FP = FP + 1
        else:
            TN = TN + 1

    f1 = (2 * TP) / (2 * TP + FP + FN)
    f1_sum += f1
    print('pre:', TP / (TP + FP), 'rec:', TP / (TP + FN))
    print('f1:', f1)

    confusion_matrix = [
        ["1=died  0=alive", "Pred class = '1'", "Pred class = '0'", 'Total actual c'],
        ["Actual class = '1'", str(TP) + "\n(TP)", str(FN) + "\n(FN)",
         str(TP + FN) + "\n(Total actual c= '1')"],
        ["Actual class = '0'", str(FP) + "\n(FP)", str(TN) + "\n(TN)",
         str(FP + TN) + "\n(Total actual c= '0')"],
        ["Total pred c", str(TP + FP) + "\n(Total pred as '1')", str(FN + TN) + "\n(Total pred as '0')",
         str(pd.concat([transactions_te_0, transactions_te_1]).shape[0])],
    ]
    print(tabulate(confusion_matrix, headers='firstrow', tablefmt='fancy_grid'))
    print(classification_report(y, pred_y, zero_division=0))

print("\n\nAvg")
print(f"Roc auc (class 1): {auc_sum / 10}")
print(f"f1: {f1_sum / 10}")

time_sec = time.time() - start_time
time_min = time_sec / 60
print("\nProcessing time of %s(): %.2f seconds (%.2f minutes)."
      % ("whole code", time.time() - start_time, time_min))
