import multiprocessing
import pandas as pd
from posneg_rule_gen.posneg_rule_generation import ponerg
from rule_gen import apriori_mlx
from util import util
import timeit

classes = None
min_support = -1
min_conf = -1
transactions_df = None
class_support_count_dict = None


@util.timeit
def classification_rule_generation(transactions, m_min_support, m_min_conf):
    global min_support
    global min_conf
    global transactions_df
    global class_support_count_dict
    global classes
    min_support = m_min_support
    min_conf = m_min_conf

    rules = []
    transactions_df = util.convert_trans_to_df(transactions)
    X_df = pd.DataFrame(transactions_df.drop(['1', '0'], axis=1))
    classes = [frozenset(['0']), frozenset(['1'])]
    class_support_count_dict = util.get_support_count_dict_df(classes, transactions_df)

    f1, previous_itemset_arr = apriori_mlx.apriori_of_size_1(X_df, min_support=min_support)
    f1 = f1.tolist()
    freq_itemsets = [f1]
    for item in f1:
        rules.extend(ponerg(item, classes, class_support_count_dict, min_conf, transactions_df))

    k = 0
    while freq_itemsets[k] is not None and len(freq_itemsets[k]) > 0:
        k_freq_itemsets, previous_itemset_arr = apriori_mlx.apriori_of_size_k(
            X_df, previous_itemset_arr, min_support=min_support, k=k + 2)
        if not k_freq_itemsets.empty:
            k_freq_itemsets = k_freq_itemsets.tolist()
            # for item in k_freq_itemsets_x_c:
            #     rules.extend(ponerg(item, c, len(transactions_per_c), min_conf, transactions_df))
            with multiprocessing.Pool() as pool:
                result = pool.map(ponerg_parallel, k_freq_itemsets)
            rules_to_extend = [x[0] for x in result if x != []]
            rules.extend(rules_to_extend)
            freq_itemsets.append(k_freq_itemsets)
        else:
            freq_itemsets.append(None)
        k += 1

    return rules


def ponerg_parallel(item):
    return ponerg(item, classes, class_support_count_dict, min_conf, transactions_df)


'''
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
'''