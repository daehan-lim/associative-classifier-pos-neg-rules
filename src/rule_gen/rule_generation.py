import multiprocessing

import pandas as pd
from src.posneg_rule_gen.posneg_rule_generation import ponerg
from src.rule_gen import apriori_mlx
from src.util import util
import itertools
import timeit

classes = []
min_support = -1
min_conf = -1
class_support_count_dict = dict()
transactions_df = None


def classification_rule_generation(transactions, m_classes, m_min_support, m_min_conf):
    global classes
    global min_support
    global min_conf
    global class_support_count_dict
    global transactions_df
    classes = m_classes
    min_support = m_min_support
    min_conf = m_min_conf
    rules = []

    transactions_df = util.convert_trans_to_df(transactions)
    itemsets_df = pd.DataFrame(transactions_df.drop(['1', '0'], axis=1))
    class_support_count_dict = util.get_support_count_dict_df(classes, transactions_df)
    f1, previous_itemset_array = apriori_mlx.apriori_of_size_1(
        itemsets_df, min_support=min_support)
    f1 = f1.tolist()
    frequent_itemsets = [f1]
    for item in f1:
        rules.extend(ponerg(item, classes, class_support_count_dict, min_conf, transactions_df))

    k = 0
    while frequent_itemsets[k] is not None and len(frequent_itemsets[k]) > 0:
        ck = _generate_ck_merge(frequent_itemsets[k], f1)
        with multiprocessing.Pool() as pool:
            result = pool.map(ponerg_parallel, ck)
        rules_to_extend = [x[0] for x in result if x != []]
        rules.extend(rules_to_extend)
        # for rule_list in result:
        #     if rule_list:
        #         rules.extend(rule_list)

        # for item in ck:
        #     rules.extend(ponerg(item, classes, class_support_count_dict, min_conf, transactions_df))
        k_freq_itemsets, previous_itemset_array = apriori_mlx.apriori_of_size_k(
            itemsets_df, previous_itemset_array, min_support=min_support, k=k + 2)
        frequent_itemsets.append(None if k_freq_itemsets.empty
                                 else k_freq_itemsets.tolist())
        k += 1

    return rules


def ponerg_parallel(ck_item):
    return ponerg(ck_item, classes, class_support_count_dict, min_conf, transactions_df)


def _greater_than_items(item_set, one_itemset_item):
    for set_item in item_set:
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
    return ck
