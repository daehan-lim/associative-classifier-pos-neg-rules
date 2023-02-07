import pandas as pd
from posneg_rule_gen.posneg_rule_generation import ponerg
from rule_gen import apriori_mlx
from util import util
import itertools
import timeit


def classification_rule_generation(transactions, classes, min_support, min_conf, corr):
    rules = []
    transactions_df = util.convert_trans_to_df(transactions)

    for c in classes:
        class_label, = c
        transactions_per_c = pd.DataFrame(
            transactions_df[transactions_df[class_label]].reset_index(drop=True).drop(['1', '0'], axis=1))
        add_rules_per_c(rules, c, transactions_per_c, min_support, min_conf, corr, transactions_df)
    return rules


def add_rules_per_c(rules, c, transactions_per_c, min_support, min_conf, corr, transactions_df):
    f1_per_c, previous_itemset_arr = apriori_mlx.apriori_of_size_1(transactions_per_c, min_support=min_support)
    f1_per_c = f1_per_c.tolist()
    for item in f1_per_c:
        rules.extend(ponerg(item, c, len(transactions_per_c), min_conf, corr, transactions_df))
    freq_itemsets = [f1_per_c]

    k = 0
    while freq_itemsets[k] is not None and len(freq_itemsets[k]) > 0:
        k_freq_itemsets_x_c, previous_itemset_arr = apriori_mlx.apriori_of_size_k(
            transactions_per_c, previous_itemset_arr, min_support=min_support, k=k + 2)
        if not k_freq_itemsets_x_c.empty:
            k_freq_itemsets_x_c = k_freq_itemsets_x_c.tolist()
            for item in k_freq_itemsets_x_c:
                rules.extend(ponerg(item, c, len(transactions_per_c), min_conf, corr, transactions_df))
            freq_itemsets.append(k_freq_itemsets_x_c)
        else:
            freq_itemsets.append(None)
        k += 1


def _greater_than_items(item_set, one_itemset_item):
    for set_item in item_set:
        if one_itemset_item <= set_item:
            return False
    return True


# def ponerg_parallel(ck_item):
#     return ponerg(ck_item, classes, class_support_count_dict, min_conf, transactions_df)


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
    return ck
