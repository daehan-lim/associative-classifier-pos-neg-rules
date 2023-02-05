import pandas as pd
from posneg_rule_gen.posneg_rule_generation import ponerg
from rule_gen import apriori_mlx
from util import util
import itertools
import timeit


def classification_rule_generation(transactions, classes, min_support, min_conf):
    rules = []

    transactions_df = util.convert_trans_to_df(transactions)
    itemsets_df = pd.DataFrame(transactions_df.drop(['1', '0'], axis=1))
    class_support_count_dict = util.get_support_count_dict_df(classes, transactions_df)
    f1, previous_itemset_array = apriori_mlx.apriori_of_size_1(itemsets_df, min_support=min_support)
    f1 = f1.tolist()
    frequent_itemsets = [f1]
    for item in f1:
        rules.extend(ponerg(item, classes, class_support_count_dict, min_conf, transactions_df))

    k = 0
    while frequent_itemsets[k] is not None and len(frequent_itemsets[k]) > 0:
        # ck = _generate_ck_merge(frequent_itemsets[k], f1)
        # with multiprocessing.Pool() as pool:
        #     rules.extend([x[0] for x in pool.map(ponerg_parallel, ck) if x != []])

        # for item in ck:
        #     rules.extend(ponerg(item, classes, class_support_count_dict, min_conf, transactions_df))
        k_freq_itemsets, previous_itemset_array = apriori_mlx.apriori_of_size_k(
            itemsets_df, previous_itemset_array, min_support=min_support, k=k + 2)
        if k_freq_itemsets.empty:
            frequent_itemsets.append(None)
        else:
            k_freq_itemsets = k_freq_itemsets.tolist()
            for item in k_freq_itemsets:
                rules.extend(ponerg(item, classes, class_support_count_dict, min_conf, transactions_df))
            frequent_itemsets.append(k_freq_itemsets)
        k += 1
    return rules


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
