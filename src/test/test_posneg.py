import csv
import timeit

from src.util import util

records = []
with open('../../data/store_data.csv', 'r') as file:
    for row in csv.reader(file):
        records.append(row)
transactions_df = util.convert_trans_to_df(records)
transactions_df

n_rows = len(records)


# itemset is a list of two items


# create a column with True if both item1 and item2 are true
def support_apriori(itemset: frozenset):
    subset = transactions_df[list(itemset)].copy()
    subset['support'] = subset.all(axis=1)
    support = subset['support'].sum()
    return support


# support_ck = util.get_support_count_dict([frozenset(['milk', 'shrimp'])], transactions=records)

# print(f'timeit = {timeit.timeit(lambda: subset['support'].sum(), number=5)}')
itemset_m = frozenset(["milk", "shrimp", "almonds", "chocolate", "beer", "sugar"])
print(util.get_item_support_count_df(itemset_m, transactions_df))
print(util.get_item_support_count(itemset_m, transactions=records))
print(timeit.timeit(lambda: support_apriori(itemset_m), number=1))
print(timeit.timeit(lambda: util.get_item_support_count(itemset_m, transactions=records), number=1))

a = 0
