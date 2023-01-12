from collections import defaultdict


def classification(itemset, rules_set, confidence_margin):
    matching_rules = []
    count = 0
    first_rule_confidence = None
    for rule in rules_set:
        if (rule['antecedent'] | frozenset([rule['consequent']])).issubset(itemset):
            if count == 0:
                count += 1
                first_rule_confidence = rule['confidence']
                matching_rules.append(rule)
            elif rule['confidence'] > first_rule_confidence - confidence_margin:
                matching_rules.append(rule)
            else:
                break

    if len(matching_rules) == 0:
        return None

    # Divide the set S into subsets based on category
    rules_by_class = defaultdict(list)
    for rule in matching_rules:
        rules_by_class[rule['consequent']].append(rule)

    # Calculate the average confidence score for each category
    avg_conf_by_group = dict()
    for rule_group in rules_by_class:
        avg_conf_by_group[rule_group] = sum(rule['confidence'] for rule in rules_by_class[rule_group]) / len(rules_by_class[rule_group])

    # Assign the new object to the class with the highest confidence score
    predicted_class = max(avg_conf_by_group.items(), key=lambda x: x[1])[0]

    return predicted_class


if __name__ == '__main__':
    my_rules = [{'antecedent': frozenset({'low fat yogurt'}), 'consequent': 'napkins', 'confidence': 0.4},
             {'antecedent': frozenset({'chocolate'}), 'consequent': 'shrimp', 'confidence': 0.251865671641791},
             {'antecedent': frozenset({'chicken'}), 'consequent': 'cream', 'confidence': 0.2857142857142857},
             {'antecedent': frozenset({'red wine'}), 'consequent': 'napkins', 'confidence': 0.2},
             {'antecedent': frozenset({'low fat yogurt', 'ground beef'}), 'consequent': 'napkins', 'confidence': 0.2},
             {'antecedent': frozenset({'green tea', 'grated cheese'}), 'consequent': 'napkins', 'confidence': 0.2},
             {'antecedent': frozenset({'frozen vegetables', 'grated cheese'}), 'consequent': 'napkins',
              'confidence': 0.2},
             {'antecedent': frozenset({'spaghetti', 'grated cheese'}), 'consequent': 'napkins', 'confidence': 0.2},
             {'antecedent': frozenset({'ground beef', 'spaghetti'}), 'consequent': 'cream',
              'confidence': 0.2857142857142857},
             {'antecedent': frozenset({'spaghetti', 'herb & pepper'}), 'consequent': 'napkins', 'confidence': 0.2},
             {'antecedent': frozenset({'pancakes', 'grated cheese'}), 'consequent': 'napkins', 'confidence': 0.2},
             {'antecedent': frozenset({'pancakes', 'butter'}), 'consequent': 'napkins', 'confidence': 0.2},
             {'antecedent': frozenset({'ground beef', 'grated cheese'}), 'consequent': 'napkins', 'confidence': 0.2},
             {'antecedent': frozenset({'grated cheese', 'butter'}), 'consequent': 'napkins', 'confidence': 0.2}]
    my_itemset = frozenset(['napkins', 'grated cheese', 'butter'])
    my_confidence_margin = 0.001

    # matching_rules = []
    # count = 0
    # first_rule_confidence = None
    # for rule in my_rules:
    #     if (rule['antecedent'] | frozenset([rule['consequent']])).issubset(my_itemset):
    #         if count == 0:
    #             count += 1
    #             first_rule_confidence = rule['confidence']
    #             matching_rules.append(rule)
    #         elif rule['confidence'] > first_rule_confidence - my_confidence_margin:
    #             matching_rules.append(rule)
    #         else:
    #             break

    # # rules_by_class = defaultdict(list)
    # #
    # # for rule in my_rules:
    # #     rules_by_class[rule['consequent']].append(rule)
    # #
    # # # Calculate the average confidence score for each category
    # # scores = dict()
    # # for rule_group in rules_by_class:
    # #     scores[rule_group] = sum(rule['confidence'] for rule in rules_by_class[rule_group]) / len(
    # #         rules_by_class[rule_group])
    #
    # # Assign the new object to the class with the highest confidence score
    # predicted_class = max(scores.items(), key=lambda x: x[1])[0]

