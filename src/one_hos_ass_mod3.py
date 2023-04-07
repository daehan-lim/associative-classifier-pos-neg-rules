import csv
import numpy as np
import copy
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import random
from mlxtend.frequent_patterns import apriori
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import math
import warnings

from rule_gen import apriori_mlx

warnings.simplefilter(action='ignore', category=FutureWarning)

result = []
for myseed in range(10):
    print('myseed', myseed)
    random.seed(myseed)
    pre = []
    with open('../data/dataset.csv', 'r') as file:
        data_set = [list(filter(None, row)) for row in csv.reader(file)]

    te = TransactionEncoder()
    te_ary = te.fit_transform(data_set)
    m_transactions = pd.DataFrame(te_ary, columns=te.columns_)

    transactions_0 = pd.DataFrame(
        m_transactions[m_transactions['0']].reset_index(drop=True).drop(['1', '0'], axis=1))
    transactions_1 = pd.DataFrame(
        m_transactions[m_transactions['1']].reset_index(drop=True).drop(['1', '0'], axis=1))

    indices = list(range(0, len(transactions_0)))
    random.shuffle(indices)
    transactions_te_0 = transactions_0.iloc[indices[:417], :]
    transactions_tr_0 = transactions_0.iloc[indices[417:], :]

    indices = list(range(0, len(transactions_1)))
    random.shuffle(indices)
    transactions_te_1 = transactions_1.iloc[indices[:43], :]
    transactions_tr_1 = transactions_1.iloc[indices[43:], :]

    transactions_tr_0_intarray = transactions_tr_0.values.astype('int')
    transactions_tr_1_intarray = transactions_tr_1.values.astype('int')

    transactions_tr = pd.concat([transactions_tr_0, transactions_tr_1])

    # frequent_items = apriori(transactions_tr, min_support=0.1)
    frequent_items = apriori_mlx.apriori(transactions_tr, min_support=0.06, low_memory=True)
    print(transactions_tr.shape, len(frequent_items))

    attributes_count = transactions_tr.shape[1]  # number of attributes
    freq_itemsets_count = len(frequent_items)  # number of frequent items
    freq_itemsets = frequent_items['itemsets']
    freq_itemsets_matrix = [list(x) for x in freq_itemsets]
    attributes_contained_in_freq_items = np.zeros((attributes_count, freq_itemsets_count))
    for i in range(len(frequent_items)):
        attributes_contained_in_freq_items[freq_itemsets_matrix[i], i] = 1

    # cls0-cls1
    freq_count_per_trans_0 = np.matmul(transactions_tr_0_intarray, attributes_contained_in_freq_items)
    freq_count_per_trans_1 = np.matmul(transactions_tr_1_intarray, attributes_contained_in_freq_items)
    # Each element in the matrix indicates the count of how many times the corresponding
    # frequent itemset occurred in the transactions of the corresponding class.

    y0 = np.concatenate((np.zeros(transactions_tr_0_intarray.shape[0]), np.ones(transactions_tr_1_intarray.shape[0])), axis=0)

    item_len = np.sum(attributes_contained_in_freq_items, axis=0)  # item length
    cls = np.zeros((freq_itemsets_count, 2))  # frequent_items['support']
    rconf = np.zeros(freq_itemsets_count)
    indc = np.zeros(freq_itemsets_count, dtype=int)
    for i in range(freq_count_per_trans_0.shape[1]):
        cls[i, 0] = (freq_count_per_trans_0[:, i] >= item_len[i]).sum() / freq_count_per_trans_0.shape[0]
        cls[i, 1] = (freq_count_per_trans_1[:, i] >= item_len[i]).sum() / freq_count_per_trans_1.shape[0]

        if cls[i, 1] <= cls[i, 0]:
            indc[i] = 0
            rconf[i] = 1 - cls[i, 1] / cls[i, 0]
        else:
            indc[i] = 1
            rconf[i] = 1 - cls[i, 0] / cls[i, 1]
        '''      
        if cls[i,0]>= cls[i,1]:
            indc[i] = 0
            rconf[i] = cls[i,0]*(cls[i,0] - cls[i,1])
        else:
            indc[i] = 1
            rconf[i] = cls[i,1]*(cls[i,1] - cls[i,0])
        '''
    print('class 0 rules: ', (indc == 0).sum(), 'class 1 rules: ', (indc == 1).sum())

    conf = copy.deepcopy(rconf)

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    sorted_indices = conf.argsort()
    sorted_indices = sorted_indices[::-1]
    bestk = 3  # Number of rules per class to leave when prunning and Number of rules per class to use

    for i in range(freq_count_per_trans_0.shape[0]):
        first0 = 0
        first1 = 0
        for j in range(freq_count_per_trans_0.shape[1]):
            if indc[sorted_indices[j]] == 0:
                if first0 < bestk and freq_count_per_trans_0[i, sorted_indices[j]] >= item_len[sorted_indices[j]]:
                    freq_count_per_trans_0[i, sorted_indices[j]] = 1
                    first0 = first0 + 1
                else:
                    freq_count_per_trans_0[i, sorted_indices[j]] = 0
            elif indc[sorted_indices[j]] == 1:
                if first1 < bestk and freq_count_per_trans_0[i, sorted_indices[j]] >= item_len[sorted_indices[j]]:
                    freq_count_per_trans_0[i, sorted_indices[j]] = 1
                    first1 = first1 + 1
                else:
                    freq_count_per_trans_0[i, sorted_indices[j]] = 0

    for i in range(freq_count_per_trans_1.shape[0]):
        first0 = 0
        first1 = 0
        for j in range(freq_count_per_trans_1.shape[1]):
            if indc[sorted_indices[j]] == 0:
                if first0 < bestk and freq_count_per_trans_1[i, sorted_indices[j]] >= item_len[sorted_indices[j]]:
                    freq_count_per_trans_1[i, sorted_indices[j]] = 1
                    first0 = first0 + 1
                else:
                    freq_count_per_trans_1[i, sorted_indices[j]] = 0
            if indc[sorted_indices[j]] == 1:
                if first1 < bestk and freq_count_per_trans_1[i, sorted_indices[j]] >= item_len[sorted_indices[j]]:
                    freq_count_per_trans_1[i, sorted_indices[j]] = 1
                    first1 = first1 + 1
                else:
                    freq_count_per_trans_1[i, sorted_indices[j]] = 0

    f1 = np.zeros(freq_itemsets_count)
    for i in range(freq_count_per_trans_0.shape[1]):
        if indc[sorted_indices[i]] == 0:
            TN = TN + np.sum(freq_count_per_trans_0[:, sorted_indices[i]])
            FN = FN + np.sum(freq_count_per_trans_1[:, sorted_indices[i]])
        else:
            FP = FP + np.sum(freq_count_per_trans_0[:, sorted_indices[i]])
            TP = TP + np.sum(freq_count_per_trans_1[:, sorted_indices[i]])
        if 2 * TP + FN + FP > 0:
            f1[i] = 2 * TP / (2 * TP + FN + FP)

    ss = math.floor(freq_itemsets_count * 0.38)
    # ss = 0
    t = np.argmax(f1[ss:])

    print('selected rules: ', ss + t + 1)
    pre.append(freq_itemsets_count)
    pre.append(ss + t + 1)

    conf = copy.deepcopy(rconf[sorted_indices[:ss + t + 1]])
    attributes_contained_in_freq_items = copy.deepcopy(attributes_contained_in_freq_items[:, sorted_indices[:ss + t + 1]])
    indc = copy.deepcopy(indc[sorted_indices[:ss + t + 1]])
    item_len = copy.deepcopy(item_len[sorted_indices[:ss + t + 1]])

    te_0_ary = transactions_te_0.values.astype('int')
    te_1_ary = transactions_te_1.values.astype('int')

    freq_count_per_test_0 = np.matmul(te_0_ary, attributes_contained_in_freq_items)
    freq_count_per_test_1 = np.matmul(te_1_ary, attributes_contained_in_freq_items)
    # Each element in the matrix indicates the count of how many times the corresponding
    # frequent itemset occurred in the transactions of the corresponding class.

    pred = np.zeros((freq_count_per_test_0.shape[0] + freq_count_per_test_1.shape[0], 2))

    ###############
    # best k rules for each class to be used
    for i in range(freq_count_per_test_0.shape[0]):

        cnt = 0
        for j in range(freq_count_per_test_0.shape[1]):
            if indc[j] == 0:
                if cnt < bestk and freq_count_per_test_0[i, j] >= item_len[j]:
                    pred[i, 0] = pred[i, 0] + conf[j]
                    cnt = cnt + 1
                elif cnt >= bestk:
                    break
        cnt = 0
        for j in range(freq_count_per_test_0.shape[1]):
            if indc[j] == 1:
                if cnt < bestk and freq_count_per_test_0[i, j] >= item_len[j]:
                    pred[i, 1] = pred[i, 1] + conf[j]
                    cnt = cnt + 1
                elif cnt >= bestk:
                    break

    for i in range(freq_count_per_test_1.shape[0]):

        cnt = 0
        for j in range(freq_count_per_test_1.shape[1]):
            if indc[j] == 0:
                if cnt < bestk and freq_count_per_test_1[i, j] >= item_len[j]:
                    pred[i + freq_count_per_test_0.shape[0], 0] = pred[i + freq_count_per_test_0.shape[0], 0] + conf[j]
                    cnt = cnt + 1
                elif cnt >= bestk:
                    break
        cnt = 0
        for j in range(freq_count_per_test_1.shape[1]):
            if indc[j] == 1:
                if cnt < bestk and freq_count_per_test_1[i, j] >= item_len[j]:
                    pred[i + freq_count_per_test_0.shape[0], 1] = pred[i + freq_count_per_test_0.shape[0], 1] + conf[j]
                    cnt = cnt + 1
                elif cnt >= bestk:
                    break

    y = np.concatenate((np.zeros(te_0_ary.shape[0]), np.ones(te_1_ary.shape[0])), axis=0)
    pred_y = np.zeros(freq_count_per_test_0.shape[0] + freq_count_per_test_1.shape[0], dtype=int)

    for i in range(pred.shape[0]):
        if pred[i, 0] >= pred[i, 1]:
            pred_y[i] = 0
        else:
            pred_y[i] = 1

    print('f1: ', f1_score(y, pred_y))
    pre.append(f1_score(y, pred_y))
    print('auc: ', roc_auc_score(y, pred[:, 1]))
    pre.append(roc_auc_score(y, pred[:, 1]))

    result.append(pre)

result.append(np.mean(result, axis=0))
table = pd.DataFrame(result)
table.to_csv('../random3.csv')
