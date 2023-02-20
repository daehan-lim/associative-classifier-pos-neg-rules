import time

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


def convert_trans_to_df(transaction):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transaction)
    data_df = pd.DataFrame(te_ary, columns=te.columns_)
    return data_df


def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


def get_item_support_count_df(itemset: frozenset, df, negated=False):
    """
    Efficient support calculation
    :param negated: Whether it should find the support of positive or negated items
    :param itemset: Items need to be in transaction
    :param df: DataFrame of transactions
    :return: support of itemset
    """
    subset = df[list(itemset)] if negated is False else ~df[list(itemset)]
    # subset['support'] = subset.all(axis=1) # returns column
    support = subset.all(axis=1).sum()
    return support


def get_support_count_not_i_and_c(itemset: frozenset, class_str, df):
    """
    Returns support of rule of type ~i and c
    """
    negated_itemset_df = ~df[list(itemset)]
    # create Series temp that says true when all the items in the antecedent are false
    temp = negated_itemset_df.all(axis=1)
    support = (temp & df[class_str]).sum()

    return support


def get_support_count_i_and_not_c(itemset: frozenset, class_str, df):
    """
    Returns support of rule of type i and ~c
    """
    itemset_df = df[list(itemset)]
    # create Series temp that says true when all the items in the antecedent are true
    temp = itemset_df.all(axis=1)
    support = (temp & ~df[class_str]).sum()

    return support


'''
def get_support_count_dict_df(ck, transactions_df):
    item_support_count = {}
    for item in ck:
        item_support_count[item] = get_item_support_count_df(item, transactions_df)
    return item_support_count


def get_support_count_dict(ck, transactions):
    item_support_count = {}
    for transaction in transactions:
        for item in ck:
            if item.issubset(transaction):
                if item not in item_support_count:
                    item_support_count[item] = 1
                else:
                    item_support_count[item] += 1
    return item_support_count



def get_item_support_count(item, transactions):
    support_count = 0
    for transaction in transactions:
        if item.issubset(transaction):
            support_count += 1
    return support_count
'''
