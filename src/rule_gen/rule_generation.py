import pandas as pd

from src.posneg_rule_gen.posneg_rule_generation import ponerg
from src.rule_gen import apriori_mlx
from src.util import util
import itertools


def classification_rule_generation(transactions, min_support, corr, min_conf):
    PCR = []
    NCR = []

    # f1 = apriori(pd.DataFrame(transactions_df), min_support=min_support, use_colnames=True, max_len=1)
    c1 = create_candidate_1(transactions)
    one_itemset_support_count = util.get_support_count(c1, transactions)
    classes = []
    for one_itemset in c1:
        one_itemset_item, = one_itemset
        classes.append(one_itemset_item)
    f1_with_support_count = create_1_freq_itemsets_with_supp(transactions, one_itemset_support_count, min_support=min_support)
    f1 = f1_with_support_count[0]
    frequent_itemsets = [f1]
    for item, support_count in zip(*f1_with_support_count):
        rules = ponerg(item, support_count, classes, one_itemset_support_count, corr, min_conf)
        PCR.extend(rules[0])
        NCR.extend(rules[1])

    transactions_df = util.convert_trans_to_df(transactions)
    k = 0
    while frequent_itemsets[k] is not None and len(frequent_itemsets[k]) > 0:
        ck = _generate_ck_merge(frequent_itemsets[k], f1)
        ck_support_count = util.get_support_count(ck, transactions)
        for item in ck:
            rules = ponerg(item, ck_support_count[item], classes,
                           one_itemset_support_count, corr, min_conf)
            PCR.extend(rules[0])
            NCR.extend(rules[1])

        k_freq_itemsets = apriori_mlx.apriori_of_size_k(pd.DataFrame(transactions_df),
                                                        min_support=min_support, use_colnames=True, k=k+2)
        frequent_itemsets.append(None if k_freq_itemsets.empty
                                 else k_freq_itemsets['itemsets'].tolist())
        k += 1
    # f1 = apriori(pd.DataFrame(transactions_df), min_support=min_support, use_colnames=True, max_len=1)
    # frequent_itemsets = [pd.DataFrame(f1)]
    # k = 1  # change to 1
    # while len(frequent_itemsets[k - 1]) > 0:
    #     # itemset_union = _merge_itemsets(frequent_itemsets[k - 1]['itemsets'], f1['itemsets'], k)
    #     c_k = util.apriori_for_transaction(itemset_union, min_support=min_support, max_len=k + 1)  # F_k-1 U f1
    #     c_k = util.apriori_of_size_k(pd.DataFrame(transactions_df), min_support=min_support, use_colnames=True, k=k+1)
    #     frequent_itemset_k = pd.DataFrame(columns=['support', 'itemsets'])
    #     for index, row in c_k.iterrows():
    #         if row['support'] >= min_support:
    #             frequent_itemset_k = frequent_itemset_k.append(row, ignore_index=True)
    #         # rules = ponerg(itemset, classes, corr, min_conf)
    #         # PCR = PCR.append(rules[0])
    #         # NCR = NCR.append(rules[1])
    #     frequent_itemsets.append(frequent_itemset_k)
    #     k += 1

    return PCR, NCR
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


def create_1_freq_itemsets_with_supp(transactions, one_itemset_support_count, min_support):
    """
    filters the candidate with the specified minimum support
    """
    # one_itemset_support_count: loop through the transaction and compute the count for each candidate (item)

    freq_itemsets = []
    support_count_list = []
    # if the support of an item is greater than the min_support, then it is considered as frequent
    for item in one_itemset_support_count:
        support_count = one_itemset_support_count[item]
        if support_count / len(transactions) >= min_support:
            freq_itemsets.append(item)
            support_count_list.append(support_count)
    return freq_itemsets, support_count_list


