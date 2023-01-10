import urllib.request
import os
import pandas as pd
from util import util
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
import itertools


def classification_rule_generation(transactions, min_support, corr, min_conf):
    PCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    NCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    # classes = transactions['class']

    f1 = apriori(pd.DataFrame(util.convert_trans_to_df(transactions)), min_support=min_support, use_colnames=True,
                 max_len=1)
    frequent_itemsets = [pd.DataFrame(f1)]
    # for item in f1:
    #     rules = ponerg(item, classes, corr, min_conf)
    #     PCR = PCR.append(rules[0])
    #     NCR = NCR.append(rules[1])

    k = 1  # change to 1
    while len(frequent_itemsets[k - 1]) > 0:
        # itemset_union = _merge_itemsets(frequent_itemsets[k - 1]['itemsets'], f1['itemsets'], k)
        # c_k = util.apriori_for_transaction(itemset_union, min_support=min_support, max_len=k + 1)  # F_k-1 U f1
        c_k = util.apriori_of_size_k(pd.DataFrame(util.convert_trans_to_df(transactions)),
                                     min_support=min_support, use_colnames=True, k=k + 1)
        frequent_itemset_k = pd.DataFrame(columns=['support', 'itemsets'])
        for index, row in c_k.iterrows():
            if row['support'] >= min_support:
                frequent_itemset_k = frequent_itemset_k.append(row, ignore_index=True)
            # rules = ponerg(itemset, classes, corr, min_conf)
            # PCR = PCR.append(rules[0])
            # NCR = NCR.append(rules[1])
        frequent_itemsets.append(frequent_itemset_k)
        k += 1
    return PCR, NCR


def ponerg(itemset, classes, corr, min_conf):
    PCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    NCR = pd.DataFrame(columns=['antecedents', 'consequents'])

    for c in classes:
        r = correlation(itemset, c)  # compute correlation between itemset and class c
        if r > corr:  # generate positive rule
            # pr = pd.DataFrame({'antecedents': rule_antecedents, 'consequents': rule_consequents})
            pr = {'antecedent': itemset, 'consequent': {c}}
            if conf(pr) >= min_conf:
                PCR.append(pr)
        elif r < -corr:  # generate negative rules
            neg_itemset = set()
            for item in itemset:
                neg_itemset.add('!' + item)
            nr1 = {'antecedent': neg_itemset, 'consequent': {c}}
            nr2 = {'antecedent': itemset, 'consequent': {'!' + c}}
            if conf(nr1) >= min_conf:
                NCR.append(nr1)
            if conf(nr2) >= min_conf:
                NCR.append(nr2)

    return PCR, NCR


def _merge_itemsets(k_itemsets, one_itemsets, k):
    # Create a list to store the resulting frozensets
    result = []
    for k_itemset in k_itemsets:
        for one_itemset in one_itemsets:
            merge = k_itemset | one_itemset
            if len(merge) == k + 1 and merge not in result:
                result.append(merge)
    # Convert the list of frozensets to a Pandas Series and return it
    return pd.Series(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # dataset = pd.read_csv('data/store_data.csv', header=None)  # To make sure the first row is not thought of as the heading
    # # Transforming the list into a list of lists, so that each transaction can be indexed easier
    # transactions = []
    # for i in range(0, dataset.shape[0]):
    #     transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

    records = []
    for line in urllib.request.urlopen("https://user.informatik.uni-goettingen.de/~sherbold/store_data.csv"):
        records.append(line.decode('ascii').strip().split(','))

    classification_rule_generation(transactions=records, min_support=0.005, min_conf=0.2,
                                   corr=1)
