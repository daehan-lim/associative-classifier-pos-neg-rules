import pandas as pd
import itertools
import src.util.ck_generation as ck_gen


def classification_rule_generation(transactions, min_support, corr, min_conf):
    PCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    NCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    # classes = transactions['class']
    c1 = ck_gen.create_candidate_1(transactions)
    f1 = ck_gen.create_freq_itemsets(transactions, c1, min_support=min_support)
    for item in f1:
        # rules = ponerg(item, classes, corr, min_conf)
        # PCR = PCR.append(rules[0])
        # NCR = NCR.append(rules[1])
        pass
    frequent_itemsets = [f1]

    k = 0
    while len(frequent_itemsets[k]) > 0:
        ck = _generate_ck_merge(frequent_itemsets[k], f1)
        for item in ck:
            # rules = ponerg(item, classes, corr, min_conf)
            # PCR = PCR.append(rules[0])
            # NCR = NCR.append(rules[1])
            pass
        k_freq_itemsets = ck_gen.create_freq_itemsets(transactions, ck, min_support=min_support)
        frequent_itemsets.append(k_freq_itemsets)
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

