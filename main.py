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
    for item in f1:
        rules = ponerg(item, classes, corr, min_conf)
        PCR = PCR.append(rules[0])
        NCR = NCR.append(rules[1])

    k = 2  # change to 1
    while len(frequent_itemsets[k - 1]) > 0:
        itemset_union_frozen = frequent_itemsets[k - 1]['itemsets'].extend(f1['itemsets'])
        itemset_union_list = [list(frozen_set) for frozen_set in itemset_union_frozen]
        c_k = util.apriori_for_transaction(itemset_union_list, min_support=0, max_len=k) # F_k-1 U f1
        for index, itemset in c_k.iterrows():
            if itemset['support'] >= min_support:
                frequent_itemsets[k] = frequent_itemsets[k].append(itemset)
            rules = ponerg(itemset, classes, corr, min_conf)
            PCR = PCR.append(rules[0])
            NCR = NCR.append(rules[1])
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


# def ponerg(transactions, min_support):
#     # Create a list of all items in the transactions
#     items = [item for sublist in transactions for item in sublist]
#     # Count the frequency of each item
#     item_counts = Counter(items)
#     # Filter out items that do not meet the minimum support
#     frequent_items = [item for item, count in item_counts.items() if count >= min_support]
#     # Sort the frequent items in ascending order by frequency
#     frequent_items.sort(key=lambda x: item_counts[x])
#
#     # Initialize the list of frequent itemsets
#     frequent_itemsets = []
#     # Generate all combinations of frequent items
#     for i in range(1, len(frequent_items) + 1):
#         for combination in itertools.combinations(frequent_items, i):
#             # Count the frequency of the combination in the transactions
#             combination_count = sum([1 for transaction in transactions if set(combination).issubset(set(transaction))])
#             # Add the combination to the list of frequent itemsets if it meets the minimum support
#             if combination_count >= min_support:
#                 frequent_itemsets.append(combination)
#     return frequent_itemsets


# Example usage
# transactions = [['A', 'B', 'C'], ['A', 'B', 'D'], ['B', 'C', 'D'], ['A', 'C']]
# min_support = 2
# frequent_itemsets = ponerg(transactions, min_support)
# print(frequent_itemsets)


# def classification_rule_generation(transactions, min_support, min_confidence):
#     # Initialize the frequent itemsets and the association rules
#     frequent_itemsets = []
#     association_rules = []
#
#     # Find the frequent itemsets
#     for i in range(1, len(transactions) + 1):
#         Ck = generate_candidate_itemsets(transactions, i)
#         for itemset in Ck:
#             if calc_support(transactions, itemset) >= min_support:
#                 frequent_itemsets.append(itemset)
#
#     # Generate the association rules from the frequent itemsets
#     for i in range(1, len(frequent_itemsets)):
#         for itemset in frequent_itemsets[i]:
#             subsets = generate_subsets(itemset)
#             for subset in subsets:
#                 confidence = calc_confidence(transactions, subset, itemset)
#                 if confidence >= min_confidence:
#                     association_rules.append((subset, itemset - subset, confidence))
#
#     return association_rules


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
