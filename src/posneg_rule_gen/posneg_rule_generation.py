import random
from src.util import util


def ponerg(itemset, classes, class_support_count_dict, corr, min_conf, transactions):
    PCR = []  # pd.DataFrame(columns=['antecedents', 'consequents'])
    NCR = []  # pd.DataFrame(columns=['antecedents', 'consequents'])

    for c in classes:
        if not c.issubset(itemset):
            r = correlation(itemset, c)  # compute correlation between itemset and class c
            c_str, = c
            if r > corr:  # generate positive rule
                # pr = pd.DataFrame({'antecedents': rule_antecedents, 'consequents': rule_consequents})
                combined_support_count = util.get_item_support_count(itemset | c, transactions)
                if (conf := confidence(combined_support_count, class_support_count_dict[c])) >= min_conf:
                    PCR.append({'antecedent': itemset, 'consequent': c_str, 'confidence': conf})
            elif r < -corr:  # generate negative rules
                neg_itemset = set()
                for item in itemset:
                    neg_itemset.add('!' + item)
                combined_support_count_nr1 = util.get_item_support_count(itemset | c, transactions)  # fix
                combined_support_count_nr2 = util.get_item_support_count(itemset | c, transactions)
                if (conf := confidence(combined_support_count_nr1, class_support_count_dict[c])) >= min_conf:
                    NCR.append({'antecedent': neg_itemset, 'consequent': c_str, 'confidence': conf})
                if (conf := confidence(combined_support_count_nr2, class_support_count_dict[c])) >= min_conf:
                    NCR.append({'antecedent': itemset, 'consequent': '!' + c_str, 'confidence': -conf})

    return PCR, NCR


def correlation(itemset, c) -> float:  # fix

    return random.uniform(-1, 1)


def confidence(combined_support_count, consequent_support_count):
    return combined_support_count / consequent_support_count
