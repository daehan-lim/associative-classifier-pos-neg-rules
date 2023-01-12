import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


def convert_trans_to_df(transaction):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transaction)
    data_df = pd.DataFrame(te_ary, columns=te.columns_)
    return data_df


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


def get_item_support_count_df(itemset: frozenset, df):
    subset = df[list(itemset)]
    # subset['support'] = subset.all(axis=1) # returns column
    support = subset.all(axis=1).sum()
    return support
