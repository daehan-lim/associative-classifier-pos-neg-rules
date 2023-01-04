import pandas as pd
from util import util
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules


def classification_rule_generation(transactions: pd.DataFrame, min_support, corr, min_conf):
    PCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    NCR = pd.DataFrame(columns=['antecedents', 'consequents'])
    classes = transactions['class']
    frequent_itemsets = [pd.DataFrame(columns=['support', 'itemsets'])]

    f1 = util.apriori_for_transaction(transactions, min_support=min_support, max_len=1)
    frequent_itemsets.append(f1)
    # for item in f1:
    #     rules = ponerg(item, classes, corr, min_conf)
    #     PCR = PCR.append(rules[0])
    #     NCR = NCR.append(rules[1])

    k = 2  # change to 1
    while len(frequent_itemsets[k - 1]) > 0:
        itemset_union_frozen = frequent_itemsets[k - 1]['itemsets'].extend(f1['itemsets'])
        itemset_union_list = [list(frozen_set) for frozen_set in itemset_union_frozen]
        c_k = util.apriori_for_transaction(itemset_union_list, min_support=0, max_len=k) # F_k-1 U f1
        for index, itemset in c_k.iterrows():
            if itemset['support'] >= min_support:
                frequent_itemsets[k] = frequent_itemsets[k].append(itemset)
            # rules = ponerg(itemset, classes, corr, min_conf)
            # PCR = PCR.append(rules[0])
            # NCR = NCR.append(rules[1])
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





# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    print('PyCharm')


