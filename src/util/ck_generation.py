import csv
import timeit
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


def apriori(transactions, min_support):
    """
    pass in the transaction data and the minimum support
    threshold to obtain the frequent itemset. Also
    store the support for each itemset, they will
    be used in the rule generation step
    """
    c1 = create_candidate_1(transactions)
    f1 = create_freq_itemsets(transactions, c1, min_support=min_support)
    k_freq_itemsets = [f1]

    k = 0
    while len(k_freq_itemsets[k]) > 0:
        freq_item = k_freq_itemsets[k]
        print(f'k = {k}, ck timeit = {timeit.timeit(lambda: create_candidate_k(freq_item, k), number=1)}')
        ck = create_candidate_k(freq_item, k)
        print(f'k = {k}, fk timeit = {timeit.timeit(lambda: create_freq_itemsets(transactions, ck, min_support=min_support), number=1)}')
        freq_item = create_freq_itemsets(transactions, ck, min_support=min_support)
        k_freq_itemsets.append(freq_item)
        k += 1

    return k_freq_itemsets


def create_freq_itemsets(transactions, ck, min_support):
    """
    filters the candidate with the specified minimum support
    """
    # loop through the transaction and compute the count for each candidate (item)
    item_support_count = {}
    for transaction in transactions:
        for item in ck:
            if item.issubset(transaction):
                if item not in item_support_count:
                    item_support_count[item] = 1
                else:
                    item_support_count[item] += 1

    freq_itemsets = []
    # if the support of an item is greater than the min_support, then it is considered as frequent
    for item in item_support_count:
        support = item_support_count[item] / len(transactions)
        if support >= min_support:
            freq_itemsets.append(item)
    return freq_itemsets


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

if __name__ == '__main__':
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
    # freq_items, item_support_dict = apriori(records, min_support=0.005)

    print(timeit.timeit(lambda: apriori(records, min_support=0.005), number=1))

    # print(f'freq items {freq_items}')
    # print(f'support dic {item_support_dict}')
