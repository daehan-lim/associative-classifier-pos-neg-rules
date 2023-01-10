import random


def ponerg(itemset, classes, corr, min_conf):
    PCR = [] # pd.DataFrame(columns=['antecedents', 'consequents'])
    NCR = [] # pd.DataFrame(columns=['antecedents', 'consequents'])

    for c in classes:
        if c not in itemset:
            r = correlation(itemset, c)  # compute correlation between itemset and class c
            if r > corr:  # generate positive rule
                # pr = pd.DataFrame({'antecedents': rule_antecedents, 'consequents': rule_consequents})
                pr = {'antecedent': itemset, 'consequent': c}
                if conf(pr) >= min_conf:
                    PCR.append(pr)
            elif r < -corr:  # generate negative rules
                neg_itemset = set()
                for item in itemset:
                    neg_itemset.add('!' + item)
                nr1 = {'antecedent': neg_itemset, 'consequent': c}
                nr2 = {'antecedent': itemset, 'consequent': '!' + c}
                if conf(nr1) >= min_conf:
                    NCR.append(nr1)
                if conf(nr2) >= min_conf:
                    NCR.append(nr2)

    return PCR, NCR


def correlation(itemset, c) -> float:
    return random.uniform(-1, 1)


def conf(rule):
    return random.uniform(0, 1)

