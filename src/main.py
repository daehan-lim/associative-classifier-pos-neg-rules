import csv
import timeit
import pandas as pd
from util import util
import itertools
import util.ck_generation as ck_gen
import rule_gen_mlx
import rule_gen_with_ck


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


if __name__ == '__main__':
    # dataset = pd.read_csv('data/store_data.csv', header=None)  # To make sure the first row is not thought of as the heading
    # for i in range(0, dataset.shape[0]):
    #     transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

    records = []
    with open('../data/store_data.csv', 'r') as file:
        for row in csv.reader(file):
            records.append(row)

    print(timeit.timeit(
        lambda: rule_gen_with_ck.classification_rule_generation(transactions=records, min_support=0.005, min_conf=0.2,
                                                                corr=1), number=1))
    # print(timeit.timeit(
    #     lambda: rule_gen_mlx.classification_rule_generation(transactions=records, min_support=0.005, min_conf=0.2, corr=1),
    #     number=5))
