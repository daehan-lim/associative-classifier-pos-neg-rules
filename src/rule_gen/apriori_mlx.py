import pandas as pd
import numpy as np


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


def apriori_of_size_1(df, min_support):
    X = df.values
    support_series = _support(X, X.shape[0], _is_sparse=False)
    ary_col_idx = np.arange(X.shape[1])
    itemset_array = ary_col_idx[support_series >= min_support].reshape(-1, 1)

    itemsets = pd.Series([frozenset(i) for i in itemset_array], dtype="object")
    mapping = {idx: item for idx, item in enumerate(df.columns)}
    itemsets = itemsets.apply(
        lambda x: frozenset([mapping[i] for i in x])
    )

    return itemsets, itemset_array


def apriori_of_size_k(df, previous_itemset_array, min_support=0.5, k=2):
    transactions_series = df.values
    rows_count = float(transactions_series.shape[0])

    combin = generate_new_combinations(previous_itemset_array)
    combin = np.fromiter(combin, dtype=np.int16)
    combin = combin.reshape(-1, k)

    if combin.size == 0:
        return pd.DataFrame(), None
    _bools = np.all(transactions_series[:, combin], axis=2)
    support_series = _support(np.array(_bools), rows_count, False)
    _mask = (support_series >= min_support).reshape(-1)
    if any(_mask):
        itemset_array = np.array(combin[_mask])
    else:
        return pd.DataFrame(), None

    itemsets = pd.Series([frozenset(i) for i in itemset_array], dtype="object")

    mapping = {idx: item for idx, item in enumerate(df.columns)}
    itemsets = itemsets.apply(
        lambda x: frozenset([mapping[i] for i in x])
    )

    return itemsets, itemset_array
