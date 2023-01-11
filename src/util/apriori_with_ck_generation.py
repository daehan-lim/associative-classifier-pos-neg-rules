import csv
import timeit

import numpy as np
import pandas as pd
from itertools import combinations


def create_candidate_1(X):
    """
    create the 1-item candidate,
    it's basically creating a frozenset for each unique item
    and storing them in a list
    """
    c1 = []
    for transaction in X:
        for t in transaction:
            t = frozenset([t])
            if t not in c1:
                c1.append(t)
    return c1


def apriori(X, min_support):
    """
    pass in the transaction data and the minimum support
    threshold to obtain the frequent itemset. Also
    store the support for each itemset, they will
    be used in the rule generation step
    """

    # the candidate sets for the 1-item is different,
    # create them independently from others
    c1 = create_candidate_1(X)
    #print(f'c1 timeit = {timeit.timeit(lambda: create_freq_item(X, create_candidate_1(X), min_support=min_support), number=1)}')
    one_freq_item, item_support_dict = create_freq_item(X, c1, min_support=min_support)
    freq_items = [one_freq_item]

    k = 0
    while len(freq_items[k]) > 0:
        freq_item = freq_items[k]
        #print(f'k = {k}, ck timeit = {timeit.timeit(lambda: create_candidate_k(freq_item, k), number=1)}')
        ck = create_candidate_k(freq_item, k)
        #print(f'k = {k}, fk timeit = {timeit.timeit(lambda: create_freq_item(X, ck, min_support=min_support), number=1)}')
        freq_item, item_support = create_freq_item(X, ck, min_support=min_support)
        freq_items.append(freq_item)
        item_support_dict.update(item_support)
        k += 1

    return freq_items, item_support_dict


def create_freq_item(X, ck, min_support):
    """
    filters the candidate with the specified
    minimum support
    """
    # loop through the transaction and compute
    # the count for each candidate (item)
    item_count = {}
    for transaction in X:
        for item in ck:
            if item.issubset(transaction):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1

    n_row = len(X)
    freq_item = []
    item_support = {}

    # if the support of an item is greater than the
    # min_support, then it is considered as frequent
    for item in item_count:
        support = item_count[item] / n_row
        if support >= min_support:
            freq_item.append(item)

        item_support[item] = support

    return freq_item, item_support


def create_candidate_k(freq_item, k):
    """create the list of k-item candidate"""
    ck = []

    # for generating candidate of size two (2-itemset)
    if k == 0:
        for f1, f2 in combinations(freq_item, 2):
            item = f1 | f2  # union of two sets
            ck.append(item)
    else:
        for f1, f2 in combinations(freq_item, 2):
            # if the two (k+1)-item sets has
            # k common elements then they will be
            # unioned to be the (k+2)-item candidate
            intersection = f1 & f2
            if len(intersection) == k:
                item = f1 | f2
                if item not in ck:
                    ck.append(item)
    return ck


def create_rules(freq_items, item_support_dict, min_confidence):
    """
    create the association rules, the rules will be a list.
    each element is a tuple of size 4, containing rules'
    left hand side, right hand side, confidence and lift
    """
    association_rules = []

    # for the list that stores the frequent items, loop through
    # the second element to the one before the last to generate the rules
    # because the last one will be an empty list. It's the stopping criteria
    # for the frequent itemset generating process and the first one are all
    # single element frequent itemset, which can't perform the set
    # operation X -> Y - X
    for idx, freq_item in enumerate(freq_items[1:(len(freq_items) - 1)]):
        for freq_set in freq_item:

            # start with creating rules for single item on
            # the right hand side
            subsets = [frozenset([item]) for item in freq_set]
            rules, right_hand_side = compute_conf(freq_items, item_support_dict,
                                                  freq_set, subsets, min_confidence)
            association_rules.extend(rules)

            # starting from 3-itemset, loop through each length item
            # to create the rules, as for the while loop condition,
            # e.g. suppose you start with a 3-itemset {2, 3, 5} then the
            # while loop condition will stop when the right hand side's
            # item is of length 2, e.g. [ {2, 3}, {3, 5} ], since this
            # will be merged into 3 itemset, making the left hand side
            # null when computing the confidence
            if idx != 0:
                k = 0
                while len(right_hand_side[0]) < len(freq_set) - 1:
                    ck = create_candidate_k(right_hand_side, k=k)
                    rules, right_hand_side = compute_conf(freq_items, item_support_dict,
                                                          freq_set, ck, min_confidence)
                    association_rules.extend(rules)
                    k += 1

    return association_rules


def compute_conf(freq_items, item_support_dict, freq_set, subsets, min_confidence):
    """
    create the rules and returns the rules info and the rules's
    right hand side (used for generating the next round of rules)
    if it surpasses the minimum confidence threshold
    """
    rules = []
    right_hand_side = []

    for rhs in subsets:
        # create the left hand side of the rule
        # and add the rules if it's greater than
        # the confidence threshold
        lhs = freq_set - rhs
        conf = item_support_dict[freq_set] / item_support_dict[lhs]
        if conf >= min_confidence:
            lift = conf / item_support_dict[rhs]
            rules_info = lhs, rhs, conf, lift
            rules.append(rules_info)
            right_hand_side.append(rhs)

    return rules, right_hand_side

# X is the transaction table from above
# we won't be using the binary format
"""
X = np.array([[1, 1, 0, 0, 0, 0],
              [1, 0, 1, 1, 1, 0],
              [0, 1, 1, 1, 0, 1],
              [1, 1, 1, 1, 0, 0],
              [1, 1, 1, 0, 0, 1]])
"""
# X = np.array([[0, 1, 0, 0],
#               [0, 2, 3, 4],
#               [1, 2, 3, 5],
#               [0, 1, 2, 3],
#               [0, 1, 2, 5]])

# dataset = pd.read_csv('../data/store_data.csv', header=None, keep_default_na=False)
# transactions = []
# for i in range(0, dataset.shape[0]):
#     transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

records = []
with open('../../data/store_data.csv', 'r') as file:
    for row in csv.reader(file):
        records.append(row)
freq_items, item_support_dict = apriori(records, min_support=0.02)

association_rules = create_rules(freq_items, item_support_dict, min_confidence = 0.05)

# print(timeit.timeit(lambda: apriori(records, min_support=0.1), number=5))

print(f'freq items {freq_items}')
print(f'support dic {item_support_dict}')
