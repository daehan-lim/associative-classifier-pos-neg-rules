import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import numpy as np


# def apriori_for_transaction(transaction, min_support, max_len=1):
#     data_df = convert_trans_to_df(transaction)
#     return apriori(pd.DataFrame(data_df), min_support=min_support, use_colnames=True,
#                    max_len=max_len)


def convert_trans_to_df(transaction):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transaction)
    data_df = pd.DataFrame(te_ary, columns=te.columns_)
    return data_df


def generate_new_combinations(old_combinations):
    """
    Generator of all combinations based on the last state of Apriori algorithm
    Parameters
    -----------
    old_combinations: np.array
        All combinations with enough support in the last step
        Combinations are represented by a matrix.
        Number of columns is equal to the combination size
        of the previous step.
        Each row represents one combination
        and contains item type ids in the ascending order
        ```
               0        1
        0      15       20
        1      15       22
        2      17       19
        ```

    Returns
    -----------
    Generator of all combinations from the last step x items
    from the previous step.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori

    """

    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        max_combination = old_combination[-1]
        mask = items_types_in_previous_step > max_combination
        valid_items = items_types_in_previous_step[mask]
        old_tuple = tuple(old_combination)
        for item in valid_items:
            yield from old_tuple
            yield item


def _support(_x, _n_rows, _is_sparse):
    out = np.sum(_x, axis=0) / _n_rows
    return np.array(out).reshape(-1)


def apriori_of_size_k(
        df, min_support=0.5, use_colnames=False, verbose=0, k=1
):
    if hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            X = df.values
        else:
            X = df.sparse.to_coo().tocsc()
        is_sparse = True
    else:
        # dense DataFrame
        X = df.values
        is_sparse = False
    support = _support(X, X.shape[0], is_sparse)
    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    rows_count = float(X.shape[0])

    all_ones = np.ones((int(rows_count), 1))

    fill_dicts_with_k_greater_than_one(X, all_ones, is_sparse, itemset_dict, k, max_itemset, min_support, rows_count,
                                       support_dict)
    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ["support", "itemsets"]
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used

    return res_df


def fill_dicts_with_k_greater_than_one(X, all_ones, is_sparse, itemset_dict, k, max_itemset, min_support, rows_count,
                                       support_dict):
    if k > 1:
        combin = generate_new_combinations(itemset_dict[max_itemset])
        combin = np.fromiter(combin, dtype=int)
        combin = combin.reshape(-1, k)

        if combin.size == 0:
            return
        if is_sparse:
            _bools = X[:, combin[:, 0]] == all_ones
            for n in range(1, combin.shape[1]):
                _bools = _bools & (X[:, combin[:, n]] == all_ones)
        else:
            _bools = np.all(X[:, combin], axis=2)

        support = _support(np.array(_bools), rows_count, is_sparse)
        _mask = (support >= min_support).reshape(-1)
        if any(_mask):
            itemset_dict[k] = np.array(combin[_mask])
            support_dict[k] = np.array(support[_mask])


# def apriori_test(
#         df, min_support=0.5, use_colnames=False, max_len=None, verbose=0
# ):
#     if hasattr(df, "sparse"):
#         # DataFrame with SparseArray (pandas >= 0.24)
#         # DataFrame with SparseArray (pandas >= 0.24)
#         if df.size == 0:
#             X = df.values
#         else:
#             X = df.sparse.to_coo().tocsc()
#         is_sparse = True
#     else:
#         # dense DataFrame
#         X = df.values
#         is_sparse = False
#     support = _support(X, X.shape[0], is_sparse)
#     ary_col_idx = np.arange(X.shape[1])
#     support_dict = {1: support[support >= min_support]}
#     itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
#     max_itemset = 1
#     rows_count = float(X.shape[0])
#
#     all_ones = np.ones((int(rows_count), 1))
#
#     while max_itemset and max_itemset < (max_len or float("inf")):
#         next_max_itemset = max_itemset + 1
#
#         combin = generate_new_combinations(itemset_dict[max_itemset])
#         combin = np.fromiter(combin, dtype=int)
#         combin = combin.reshape(-1, next_max_itemset)
#
#         if combin.size == 0:
#             break
#         if verbose:
#             print(
#                 "\rProcessing %d combinations | Sampling itemset size %d"
#                 % (combin.size, next_max_itemset),
#                 end="",
#             )
#
#         if is_sparse:
#             _bools = X[:, combin[:, 0]] == all_ones
#             for n in range(1, combin.shape[1]):
#                 _bools = _bools & (X[:, combin[:, n]] == all_ones)
#         else:
#             _bools = np.all(X[:, combin], axis=2)
#
#         support = _support(np.array(_bools), rows_count, is_sparse)
#         _mask = (support >= min_support).reshape(-1)
#         if any(_mask):
#             itemset_dict[next_max_itemset] = np.array(combin[_mask])
#             support_dict[next_max_itemset] = np.array(support[_mask])
#             max_itemset = next_max_itemset
#         else:
#             # Exit condition
#             break
#
#     all_res = []
#     for k in sorted(itemset_dict):
#         support = pd.Series(support_dict[k])
#         itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")
#
#         res = pd.concat((support, itemsets), axis=1)
#         all_res.append(res)
#
#     res_df = pd.concat(all_res)
#     res_df.columns = ["support", "itemsets"]
#     if use_colnames:
#         mapping = {idx: item for idx, item in enumerate(df.columns)}
#         res_df["itemsets"] = res_df["itemsets"].apply(
#             lambda x: frozenset([mapping[i] for i in x])
#         )
#     res_df = res_df.reset_index(drop=True)
#
#     if verbose:
#         print()  # adds newline if verbose counter was used
#
#     return res_df
