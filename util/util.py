import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder


def apriori_for_transaction(transaction, min_support=0, max_len=1):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transaction)
    data_df = pd.DataFrame(te_ary, columns=te.columns_)
    return apriori(pd.DataFrame(data_df), min_support=min_support, use_colnames=True,
                   max_len=max_len)