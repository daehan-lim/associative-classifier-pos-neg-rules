import csv
import math
import timeit

from src.util import util

records = []
with open('../../data/eicu_three_meds.csv', 'r') as file:
    for row in csv.reader(file):
        records.append(row)
transactions_df = util.convert_trans_to_df(records)
transactions_df

n_rows = len(records)
X = frozenset(['2 ML  -  METOCLOPRAMIDE HCL 5 MG/ML IJ SOLN', '3 ML VIAL : INSULIN LISPRO (HUMAN) 100 UNIT/ML SC SOLN'])
Y = frozenset(['Alive'])


def correlation():
    f11 = util.get_item_support_count_df(X | Y, transactions_df)  # given
    fm1 = util.get_item_support_count_df(Y, transactions_df)  # given by class_support_count_dict[c]
    f1m = util.get_item_support_count_df(X, transactions_df)
    f00 = util.get_item_support_count_df(X | Y, transactions_df, negated=True)
    f01 = fm1 - f11
    f10 = f1m - f11
    fm0 = f10 + f00
    f0m = f01 + f00
    coor = (f11 * f00 - f10 * f01) / math.sqrt(fm0 * fm1 * f1m * f0m)


correlation()


# f11 the number of times X and Y appear together in the same transaction (support of X and Y)
# f01 the number of transactions that contain Y but not X (support of !X and Y)
# f10 the number of transactions that contain X but not Y (support of X and !Y)
# f00 the number of times that neither X nor Y appear in the same transaction (support of !X and !Y)
# f1+ support count for X
# f+1 support count for Y
# f+0 = f10 + f00
# f0+ = f01 + f00


# create a column with True if both item1 and item2 are true
def support_apriori(itemset: frozenset):
    subset = transactions_df[list(itemset)].copy()
    subset['support'] = subset.all(axis=1)
    support = subset['support'].sum()
    return support


# support_ck = util.get_support_count_dict([frozenset(['milk', 'shrimp'])], transactions=records)

# print(f'timeit = {timeit.timeit(lambda: subset['support'].sum(), number=5)}')
itemset_m = frozenset(['2 ML  -  METOCLOPRAMIDE HCL 5 MG/ML IJ SOLN', '3 ML VIAL : INSULIN LISPRO (HUMAN) 100 UNIT/ML SC SOLN'])
print(util.get_item_support_count_df(itemset_m, transactions_df))
print(util.get_item_support_count(itemset_m, transactions=records))
print(timeit.timeit(lambda: support_apriori(itemset_m), number=1))
print(timeit.timeit(lambda: util.get_item_support_count(itemset_m, transactions=records), number=1))

a = 0
