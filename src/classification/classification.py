from collections import defaultdict


# First rule that matches object
def predict_proba(object_o, rules_set):
    for rule in rules_set:
        if (rule['antecedent']).issubset(object_o):
            return rule['confidence']
    return -1

# First rule that matches object
def classify(object_o, rules_set, confidence_margin):
    for rule in rules_set:
        if (rule['antecedent']).issubset(object_o):
            return int(rule['consequent']), rule['confidence']
    return -1, 0

    # # Using confidence margin
    # def classify(object_o, rules_set, confidence_margin):
    #     matching_rules = []
    #     count = 0
    #     first_rule_confidence = None
    #     for rule in rules_set:
    #         if (rule['antecedent']).issubset(object_o):
    #             if count == 0:
    #                 count += 1
    #                 first_rule_confidence = rule['confidence']
    #                 matching_rules.append(rule)
    #             elif abs(rule['confidence']) > first_rule_confidence - confidence_margin:
    #                 matching_rules.append(rule)
    #             else:
    #                 break
    #
    #     if len(matching_rules) == 0:
    #         return -1
    #
    # # Divide the set S into subsets based on category
    # rules_by_class = defaultdict(list)
    # for rule in matching_rules:
    #     rules_by_class[rule['consequent'].replace('!', '')].append(rule)
    #
    # # Calculate the average confidence score for each category
    # avg_conf_by_group = dict()
    # for rule_group in rules_by_class:
    #     avg_conf_by_group[rule_group] = sum(rule['confidence'] for rule in rules_by_class[rule_group]) / len(
    #         rules_by_class[rule_group])
    #
    # predicted_class = -1
    # # Assign the new object to the class with the highest confidence score
    # if (max_confidence := max(avg_conf_by_group.values())) > 0:
    #     predicted_class = [c for c, conf in avg_conf_by_group.items() if conf == max_confidence][0]
    #
    # return int(predicted_class)

# #Every rule that matches object
# def classification(object_o, rules_set, confidence_margin):
#     matching_rules = [rule for rule in rules_set if (rule['antecedent']).issubset(object_o)]
#     if len(matching_rules) == 0:
#         return 0
#
#     # Divide the set S into subsets based on category
#     rules_by_class = defaultdict(list)
#     for rule in matching_rules:
#         rules_by_class[rule['consequent'].replace('!', '')].append(rule)
#
#     # Calculate the average confidence score for each category
#     avg_conf_by_group = dict()
#     for rule_group in rules_by_class:
#         avg_conf_by_group[rule_group] = sum(rule['confidence'] for rule in rules_by_class[rule_group]) / len(
#             rules_by_class[rule_group])
#
#     predicted_class = 0
#     # Assign the new object to the class with the highest confidence score
#     if (max_confidence := max(avg_conf_by_group.values())) > 0:
#         predicted_class = [c for c, conf in avg_conf_by_group.items() if conf == max_confidence][0]
#
#     return int(predicted_class)
