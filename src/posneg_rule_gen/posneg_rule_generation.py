import math
from util import util
import timeit


def ponerg(itemset, classes, class_supp_count_dict, min_conf, transactions_df):
    rules = []
    for c in classes:
        i_and_c_supp_count = util.get_item_support_count_df(itemset | c, transactions_df)
        i_supp_count = util.get_item_support_count_df(itemset, transactions_df)
        lift = get_lift(i_and_c_supp_count, i_supp_count, class_supp_count_dict[c], len(transactions_df))
        c_str, = c
        if lift > 1:
            if (conf := confidence(i_and_c_supp_count, i_supp_count)) >= min_conf:
                rules.append({'antecedent': itemset, 'consequent': c_str, 'confidence': conf})
            break
    return rules


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
    return (f11 * f00 - f10 * f01) / math.sqrt(f_plus_0 * f_plus_1 * f1_plus * f0_plus)


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
