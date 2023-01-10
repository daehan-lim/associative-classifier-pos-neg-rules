import csv
import timeit
import urllib.request
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules


records = []
with open('data/store_data.csv', 'r') as file:
    for row in csv.reader(file):
        records.append(row)

# we first need to create a one-hot encoding of our transactions
te = TransactionEncoder()
te_ary = te.fit_transform(records)
data_df = pd.DataFrame(te_ary, columns=te.columns_)

# frequent_itemsets = apriori(pd.DataFrame(data_df), min_support=0.1, use_colnames=True)
print(timeit.timeit(lambda: apriori(pd.DataFrame(data_df), min_support=0.005, use_colnames=True), number=5))
# print(f'{frequent_itemsets}')

# rules = association_rules(frequent_itemsets, metric="confidence",
#                   min_threshold=0.5).sort_values('lift', ascending=False)
# print(rules)
