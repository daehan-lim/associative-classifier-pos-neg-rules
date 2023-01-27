from collections import defaultdict


def classification(object_o, rules_set):
    matching_rules = [rule for rule in rules_set if rule_matches_object(rule, object_o)]
    if len(matching_rules) == 0:
        return None

    # Divide the set S into subsets based on category
    rules_by_class = defaultdict(list)
    for rule in matching_rules:
        rules_by_class[rule['consequent'].replace('!', '')].append(rule)

    # Calculate the average confidence score for each category
    avg_conf_by_group = dict()
    for rule_group in rules_by_class:
        avg_conf_by_group[rule_group] = sum(rule['confidence'] for rule in rules_by_class[rule_group]) / len(
            rules_by_class[rule_group])

    predicted_class = None
    # Assign the new object to the class with the highest confidence score
    if (max_confidence := max(avg_conf_by_group.values())) > 0:
        predicted_class = [c for c, conf in avg_conf_by_group.items() if conf == max_confidence][0]

    # predicted_class = max(avg_conf_by_group.items(), key=lambda x: x[1])[0]
    return predicted_class


def rule_matches_object(rule, object_o: frozenset) -> bool:
    if next(iter(rule['antecedent']))[0] == '!':  # says if rule is of the type ~X -> C
        # Return whether it's true none of the items in the antecedent are contained in the object (transaction)
        return object_o.isdisjoint(frozenset([item.replace('!', '') for item in rule['antecedent']]))
    return (rule['antecedent']).issubset(object_o)


if __name__ == '__main__':
    pass
    # my_rules = [{'antecedent': frozenset({'low fat yogurt'}), 'consequent': 'napkins', 'confidence': 0.4},
    #             {'antecedent': frozenset({'chocolate'}), 'consequent': 'shrimp', 'confidence': 0.251865671641791},
    #             {'antecedent': frozenset({'chicken'}), 'consequent': 'cream', 'confidence': 0.2857142857142857},
    #             {'antecedent': frozenset({'red wine'}), 'consequent': 'napkins', 'confidence': 0.2},
    #             {'antecedent': frozenset({'low fat yogurt', 'ground beef'}), 'consequent': 'napkins',
    #              'confidence': 0.2},
    #             {'antecedent': frozenset({'green tea', 'grated cheese'}), 'consequent': 'napkins', 'confidence': 0.2},
    #             {'antecedent': frozenset({'frozen vegetables', 'grated cheese'}), 'consequent': 'napkins',
    #              'confidence': 0.2},
    #             {'antecedent': frozenset({'spaghetti', 'grated cheese'}), 'consequent': 'napkins', 'confidence': 0.2},
    #             {'antecedent': frozenset({'ground beef', 'spaghetti'}), 'consequent': 'cream',
    #              'confidence': 0.2857142857142857},
    #             {'antecedent': frozenset({'spaghetti', 'herb & pepper'}), 'consequent': 'napkins', 'confidence': 0.2},
    #             {'antecedent': frozenset({'pancakes', 'grated cheese'}), 'consequent': 'napkins', 'confidence': 0.2},
    #             {'antecedent': frozenset({'pancakes', 'butter'}), 'consequent': 'napkins', 'confidence': 0.2},
    #             {'antecedent': frozenset({'ground beef', 'grated cheese'}), 'consequent': 'napkins', 'confidence': 0.2},
    #             {'antecedent': frozenset({'grated cheese', 'butter'}), 'consequent': 'napkins', 'confidence': 0.2}]
    # my_itemset = frozenset(['napkins', 'grated cheese', 'butter'])
    # my_confidence_margin = 0.001

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

    # rules_by_class = defaultdict(list)
    #
    # matching_rules = [
    #     {'antecedent': frozenset({'chocolate'}), 'consequent': '!shrimp', 'confidence': 0.251865671641791},
    #     {'antecedent': frozenset({'almond'}), 'consequent': 'shrimp', 'confidence': 0.351865671641791},
    #     {'antecedent': frozenset({'chicken'}), 'consequent': '!shrimp', 'confidence': -0.2857142857142857}
    # ]
    # for rule in matching_rules:
    #     rules_by_class[rule['consequent'].replace('!', '')].append(rule)
    #
    # # Calculate the average confidence score for each category
    # scores = dict()
    # for rule_group in rules_by_class:
    #     # all('!' in rule['consequent'] for rule in rules_by_class[rule_group])
    #     scores[rule_group] = sum(rule['confidence'] for rule in rules_by_class[rule_group]) / len(
    #         rules_by_class[rule_group])
    #
    # if (max_score := max(scores.values())) > 0:
    #     predicted_class = [c for c, score in scores.items() if score == max_score][0]
    # else:
    #     predicted_class = None
    #
    # # Assign the new object to the class with the highest confidence score
    # # predicted_class = max(scores.items(), key=lambda x: x[1])[0]
    # predicted_class
