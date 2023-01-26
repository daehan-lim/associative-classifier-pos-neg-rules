import pandas as pd

from src.posneg_rule_gen.posneg_rule_generation import ponerg
from src.rule_gen import apriori_mlx
from src.util import util
import itertools
from mlxtend.preprocessing import TransactionEncoder
import timeit


def classification_rule_generation(transactions, classes, min_support, corr, min_conf):
    PCR = []
    NCR = []

    transactions_df = util.convert_trans_to_df(transactions)
    itemsets_df = pd.DataFrame(transactions_df.drop(['1', '0'], axis=1))
    class_support_count_dict = util.get_support_count_dict_df(classes, transactions_df)
    f1, previous_itemset_array = apriori_mlx.apriori_of_size_1(
        itemsets_df, min_support=min_support)
    f1 = f1.tolist()
    frequent_itemsets = [f1]
    for item in f1:
        rules = ponerg(item, classes, class_support_count_dict, corr, min_conf, transactions_df)
        PCR.extend(rules[0])
        NCR.extend(rules[1])

    k = 0
    while frequent_itemsets[k] is not None and len(frequent_itemsets[k]) > 0:
        ck = _generate_ck_merge(frequent_itemsets[k], f1)
        for item in ck:
            rules = ponerg(item, classes, class_support_count_dict, corr, min_conf, transactions_df)
            PCR.extend(rules[0])
            NCR.extend(rules[1])

        k_freq_itemsets, previous_itemset_array = apriori_mlx.apriori_of_size_k(
            itemsets_df, previous_itemset_array,
            min_support=min_support, k=k+2)
        frequent_itemsets.append(None if k_freq_itemsets.empty
                                 else k_freq_itemsets.tolist())
        k += 1

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


def create_candidate_1(transactions, classes):
    """
    create the 1-item candidate,
    it's basically creating a frozenset for each unique item
    and storing them in a list
    """
    c1 = []
    for transaction in transactions:
        for t in transaction:
            if t not in classes:
                t = frozenset([t])
                if t not in c1:
                    c1.append(t)
    return c1


