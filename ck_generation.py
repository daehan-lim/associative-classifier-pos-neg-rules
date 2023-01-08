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
    freq_item, item_support_dict = create_freq_item(X, c1, min_support=0.5)
    freq_items = [freq_item]

    k = 0
    while len(freq_items[k]) > 0:
        freq_item = freq_items[k]
        ck = create_candidate_k(freq_item, k)
        freq_item, item_support = create_freq_item(X, ck, min_support=0.5)
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

    n_row = X.shape[0]
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


# X is the transaction table from above
# we won't be using the binary format
"""
X = np.array([[1, 1, 0, 0, 0, 0],
              [1, 0, 1, 1, 1, 0],
              [0, 1, 1, 1, 0, 1],
              [1, 1, 1, 1, 0, 0],
              [1, 1, 1, 0, 0, 1]])
"""
X = np.array([[0, 1],
              [0, 2, 3, 4],
              [1, 2, 3, 5],
              [0, 1, 2, 3],
              [0, 1, 2, 5]])
freq_items, item_support_dict = apriori(X, min_support = 0.5)
freq_items

