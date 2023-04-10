import math
from util import util


def ponerg(itemset, classes, class_supp_count_dict, transactions_df):
    rules = []
    cls0 = util.get_item_support_count_df(itemset | classes[0], transactions_df) / class_supp_count_dict[classes[0]]
    cls1 = util.get_item_support_count_df(itemset | classes[1], transactions_df) / class_supp_count_dict[classes[1]]
    if cls0 >= cls1:
        rules.append({'antecedent': itemset, 'consequent': '0', 'confidence': 1 - cls1 / cls0})
    else:
        rules.append({'antecedent': itemset, 'consequent': '1', 'confidence': 1 - cls0 / cls1})
    # for c in classes:
    #     c_str, = c
    #     # i_and_c_supp_count = util.get_item_support_count_df(itemset | c, transactions_df)
    #     # i_supp_count = util.get_item_support_count_df(itemset, transactions_df)
    #     # r = correlation(itemset, c, transactions_df, i_supp_count, i_and_c_supp_count,
    #     #                 class_supp_count_dict[c])
    #     # if (conf := r) > 0.3:
    #     cls = util.get_item_support_count_df(itemset | c, transactions_df) / class_supp_count_dict[c]
    #     p = util.get_item_support_count_df(itemset, transactions_df) / len(transactions_df)
    #     if (conf := cls / p) > 1:  # same as get_lift
    #         rules.append({'antecedent': itemset, 'consequent': c_str, 'confidence': conf})
    #         break
    return rules
    # if lift > 1:
    #     i_and_not_c_supp_count = util.get_support_count_i_and_not_c(itemset, c_str, transactions_df)
    #     not_c_supp_count = util.get_item_support_count_df(c, transactions_df, negated=True)
    #     # if (conf := confidence_selection(i_and_c_supp_count, i_supp_count, class_supp_count_dict[c],
    #     #                                  i_and_not_c_supp_count, not_c_supp_count)) >= min_conf:
    #     if (conf := confidence(i_and_c_supp_count, i_supp_count)) >= min_conf:
    #         rules.append({'antecedent': itemset, 'consequent': c_str, 'confidence': conf})


def get_lift(i_and_c_supp_count, i_supp_count, class_supp_count, N):
    if i_supp_count == 0 or class_supp_count == 0:
        return 0
    return (i_and_c_supp_count * N) / (i_supp_count * class_supp_count)


def correlation(itemset, c, transactions_df, f1_plus, f11, f_plus_1):
    f00 = util.get_item_support_count_df(itemset | c, transactions_df, negated=True)

    f01 = f_plus_1 - f11
    f10 = f1_plus - f11
    f_plus_0 = f10 + f00
    f0_plus = f01 + f00

    # f_01 = util.get_support_count_not_i_and_c(itemset, list(c)[0], transactions_df)
    # f_10 = util.get_support_count_i_and_not_c(itemset, list(c)[0], transactions_df)
    # matrix = [
    #     [f11, f10, f1_plus],
    #     [f01, f00, f0_plus],
    #     [f_plus_1, f_plus_0, len(transactions_df.index)],
    # ]

    if f_plus_0 * f_plus_1 * f1_plus * f0_plus == 0:
        return 0
    N = len(transactions_df)
    # corr1 = (f11 * f00 - f10 * f01) / math.sqrt(f_plus_0 * f_plus_1 * f1_plus * f0_plus)
    # corr2 =  (N * f11 - f1_plus * f_plus_1) / math.sqrt(f_plus_0 * f_plus_1 * f1_plus * f0_plus)
    corr = (N * f11 - f1_plus * f_plus_1) / math.sqrt(f1_plus * (N - f1_plus) * f_plus_1 * (N - f_plus_1))
    return corr

# f11 the number of times X and Y appear together in the same transaction (support of X and Y)
# f01 the number of transactions that contain Y but not X (support of !X and Y)
# f10 the number of transactions that contain X but not Y (support of X and !Y)
# f00 the number of times that neither X nor Y appear in the same transaction (support of !X and !Y)
# f1+ support count for X
# f+1 support count for Y
# f+0 = f10 + f00
# f0+ = f01 + f00


def confidence(i_and_c_supp_count, i_supp_count):
    return i_and_c_supp_count / i_supp_count


def confidence_selection(i_and_c_supp_count, i_supp_count, class_supp_count, i_and_not_c_supp_count, not_c_supp_count):
    if class_supp_count == 0 or not_c_supp_count == 0 or i_and_not_c_supp_count == 0:
        return 0
    conf = confidence(i_and_c_supp_count, i_supp_count)
    class_supp = i_and_c_supp_count / class_supp_count
    css = i_and_not_c_supp_count / not_c_supp_count
    # result = not_i_and_not_c_supp_count / i_and_not_c_supp_count #ML
    result = conf * (class_supp / css)
    # result = class_supp / css
    return result
