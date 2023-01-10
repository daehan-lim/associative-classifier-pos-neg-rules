import pandas as pd
from src.rule_gen import apriori_mlx
from src.util import util
import itertools


def classification_rule_generation(transactions, min_support, corr, min_conf):
    PCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    NCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    # classes = transactions['class']

    # f1 = apriori(pd.DataFrame(transactions_df), min_support=min_support, use_colnames=True, max_len=1)
    c1 = create_candidate_1(transactions)
    f1 = create_freq_itemsets(transactions, c1, min_support=min_support)
    frequent_itemsets = [f1]
    for item in f1:
        # rules = ponerg(item, classes, corr, min_conf)
        # PCR = PCR.append(rules[0])
        # NCR = NCR.append(rules[1])
        pass

    transactions_df = util.convert_trans_to_df(transactions)
    k = 0
    while frequent_itemsets[k] is not None and len(frequent_itemsets[k]) > 0:
        ck = _generate_ck_merge(frequent_itemsets[k], f1)
        for item in ck:
            # rules = ponerg(item, classes, corr, min_conf)
            # PCR = PCR.append(rules[0])
            # NCR = NCR.append(rules[1])
            pass

        k_freq_itemsets = apriori_mlx.apriori_of_size_k(pd.DataFrame(transactions_df),
                                                        min_support=min_support, use_colnames=True, k=k+2)
        frequent_itemsets.append(None if k_freq_itemsets.empty
                                 else k_freq_itemsets['itemsets'].tolist())
        k += 1

    return frequent_itemsets
    # return PCR, NCR


def _greater_than_items(item_set, one_itemset_item):
    for set_item in sorted(item_set):
        if one_itemset_item <= set_item:
            return False
    return True


def _generate_ck_merge(k_freq_itemsets, one_freq_itemsets):
    # Create a list to store the resulting frozensets
    ck = []
    # for generating candidate of size two (2-itemset)
    if k_freq_itemsets == one_freq_itemsets:
        for f1, f2 in itertools.combinations(one_freq_itemsets, 2):
            item = f1 | f2  # union of two sets
            ck.append(item)
    else:
        for one_itemset in one_freq_itemsets:
            for k_itemset in k_freq_itemsets:
                one_itemset_item, = one_itemset  # unpacking the only element in set
                if _greater_than_items(k_itemset, one_itemset_item):
                    ck.append(k_itemset | one_itemset)
    # Convert the list of frozensets to a Pandas Series and return it
    return ck


def create_candidate_1(transactions):
    """
    create the 1-item candidate,
    it's basically creating a frozenset for each unique item
    and storing them in a list
    """
    c1 = []
    for transaction in transactions:
        for t in transaction:
            t = frozenset([t])
            if t not in c1:
                c1.append(t)
    return c1


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
