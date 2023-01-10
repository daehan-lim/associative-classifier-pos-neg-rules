import csv
import timeit
import pandas as pd
from util import util
import itertools

import util.ck_generation as ck_gen


def classification_rule_generation(transactions, min_support, corr, min_conf):
    PCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    NCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    # classes = transactions['class']

    transactions_df = util.convert_trans_to_df(transactions)
    # f1 = apriori(pd.DataFrame(transactions_df), min_support=min_support, use_colnames=True, max_len=1)
    # f1 = ck_gen.create_freq_item(transactions, ck_gen.create_candidate_1(transactions), min_support=min_support)[0]
    # frequent_itemsets = [pd.DataFrame(f1)]
    # for item in f1:
    #     rules = ponerg(item, classes, corr, min_conf)
    #     PCR = PCR.append(rules[0])
    #     NCR = NCR.append(rules[1])

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

    # k = 1  # change to 1
    # while len(frequent_itemsets[k - 1]) > 0:
    #     # itemset_union = _merge_itemsets(frequent_itemsets[k - 1]['itemsets'], f1['itemsets'], k)
    #     c_k = util.apriori_for_transaction(itemset_union, min_support=min_support, max_len=k + 1)  # F_k-1 U f1
    #     c_k = util.apriori_of_size_k(pd.DataFrame(transactions_df), min_support=min_support, use_colnames=True, k=k+1)
    #     frequent_itemset_k = pd.DataFrame(columns=['support', 'itemsets'])
    #     for index, row in c_k.iterrows():
    #         if row['support'] >= min_support:
    #             frequent_itemset_k = frequent_itemset_k.append(row, ignore_index=True)
    #         # rules = ponerg(itemset, classes, corr, min_conf)
    #         # PCR = PCR.append(rules[0])
    #         # NCR = NCR.append(rules[1])
    #     frequent_itemsets.append(frequent_itemset_k)
    #     k += 1

    return frequent_itemsets
    # return PCR, NCR


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # dataset = pd.read_csv('data/store_data.csv', header=None)  # To make sure the first row is not thought of as the heading
    # # Transforming the list into a list of lists, so that each transaction can be indexed easier
    # transactions = []
    # for i in range(0, dataset.shape[0]):
    #     transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

    # records = []
    # for line in urllib.request.urlopen("https://user.informatik.uni-goettingen.de/~sherbold/store_data.csv"):
    #     records.append(line.decode('ascii').strip().split(','))

    records = []
    with open('../data/store_data.csv', 'r') as file:
        for row in csv.reader(file):
            records.append(row)

    # print(classification_rule_generation(transactions=records, min_support=0.02, min_conf=0.2, corr=1))
    print(timeit.timeit(lambda: classification_rule_generation(transactions=records, min_support=0.02, min_conf=0.2, corr=1), number=1))
