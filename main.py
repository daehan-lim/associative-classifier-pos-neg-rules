import os
import urllib.request

import pandas as pd
from util import util
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules


def classification_rule_generation(transactions, min_support, corr, min_conf):
    PCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    NCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    # classes = transactions['class']

    f1 = util.apriori_for_transaction(transactions, min_support=min_support, max_len=1)
    frequent_itemsets = [pd.DataFrame(f1)]
    # for item in f1:
    #     rules = ponerg(item, classes, corr, min_conf)
    #     PCR = PCR.append(rules[0])
    #     NCR = NCR.append(rules[1])

    k = 1  # change to 1
    while len(frequent_itemsets[k - 1]) > 0:
        itemset_union_frozen = frequent_itemsets[k - 1]['itemsets'].append(f1['itemsets'])
        itemset_union_list = [list(frozen_set) for frozen_set in itemset_union_frozen]
        c_k = util.apriori_for_transaction(itemset_union_list, min_support=min_support, max_len=k + 1) # F_k-1 U f1
        for index, itemset in c_k.iterrows():
            if itemset['support'] >= min_support:
                frequent_itemsets[k] = frequent_itemsets[k].append(itemset)
            # rules = ponerg(itemset, classes, corr, min_conf)
            # PCR = PCR.append(rules[0])
            # NCR = NCR.append(rules[1])
        k += 1
    return PCR, NCR




# def ponerg(itemset, classes, corr, min_conf):
#     PCR = pd.DataFrame(columns=['antecedents', 'consequents'])
#     NCR = pd.DataFrame(columns=['antecedents', 'consequents'])
#
#     for c in classes:
#         r = correlation(itemset, c)  # compute correlation between itemset and class c
#         if r > corr:  # generate positive rule
#             # pr = pd.DataFrame({'antecedents': rule_antecedents, 'consequents': rule_consequents})
#             pr = {'antecedent': itemset, 'consequent': {c}}
#             if conf(pr) >= min_conf:
#                 PCR.append(pr)
#         elif r < -corr:  # generate negative rules
#             neg_itemset = set()
#             for item in itemset:
#                 neg_itemset.add('!' + item)
#             nr1 = {'antecedent': neg_itemset, 'consequent': {c}}
#             nr2 = {'antecedent': itemset, 'consequent': {'!' + c}}
#             if conf(nr1) >= min_conf:
#                 NCR.append(nr1)
#             if conf(nr2) >= min_conf:
#                 NCR.append(nr2)
#
#     return PCR, NCR





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # dataset = pd.read_csv('data/store_data.csv', header=None)  # To make sure the first row is not thought of as the heading
    # # Transforming the list into a list of lists, so that each transaction can be indexed easier
    # transactions = []
    # for i in range(0, dataset.shape[0]):
    #     transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

    records = []
    for line in urllib.request.urlopen("https://user.informatik.uni-goettingen.de/~sherbold/store_data.csv"):
        # this also means we need to decode the binary string into ascii
        records.append(line.decode('ascii').strip().split(','))

    classification_rule_generation(transactions=records, min_support=0.005, min_conf=0.2,
                                   corr=1)


    print('PyCharm')


