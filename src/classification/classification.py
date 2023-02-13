from collections import defaultdict


def classification(object_o, rules_set, confidence_margin):
    for rule in rules_set:
        if (rule['antecedent']).issubset(object_o):
            return int(rule['consequent'])
    return 0

