import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


def convert_trans_to_df(transaction):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transaction)
    data_df = pd.DataFrame(te_ary, columns=te.columns_)
    return data_df


def get_support_count(ck, transactions):
    item_support_count = {}
    for transaction in transactions:
        for item in ck:
            if item.issubset(transaction):
                if item not in item_support_count:
                    item_support_count[item] = 1
                else:
                    item_support_count[item] += 1
    return item_support_count
