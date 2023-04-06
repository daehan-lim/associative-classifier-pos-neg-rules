import random
import re

import pandas as pd
import csv
import numpy as np
from util import util
from pathlib import Path

with open('../../data/dataset.csv', 'r') as file:
    dataset = [list(filter(None, row)) for row in csv.reader(file)]

for seed in range(1, 10):
    random.seed(seed)
    transactions_df = util.convert_trans_to_df(dataset)
    indices = list(range(0, len(dataset)))
    random.shuffle(indices)
    # training_set = transactions_df.iloc[indices[460:], :].reset_index(drop=True)
    test_set = transactions_df.iloc[indices[:460], :].reset_index(drop=True)

    arr = np.arange(transactions_df.values.shape[1])
    new_column_names = {col: str(i - 1) for i, col in enumerate(transactions_df.columns)}
    new_column_names['0'] = 148
    new_column_names['1'] = 149

    transactions_df.rename(columns=new_column_names, inplace=True)
    transactions_df.rename(columns=new_column_names, inplace=True)

    transactions = transactions_df.drop([148, 149], axis=1).apply(lambda row: row.index[row], axis=1).tolist()
    transactions = pd.DataFrame(transactions)
    transactions.fillna("", inplace=True)

    y = transactions_df.apply(lambda row: 148 if row[148] else 149, axis=1).tolist()
    transactions['class'] = y

    filename = '../../data/dataset_cpar' + str(seed)
    filepath = Path(filename + '.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(filepath, index=False, header=False)

    filename = '../../data/dataset_cpar' + str(seed)
    text = open(filename + '.csv', "r")
    text = ''.join([i for i in text])
    text = re.sub(r',,+', ',', text)
    text = re.sub(r',', ' ', text)
    x = open(filename + '.num', "w")
    x.writelines(text)
    x.close()

    # test_filename = '../../data/test_cba' + str(seed)
    # text_test = open(test_filename + '.csv', "r")
    # text_test = ''.join([i for i in text_test])
    # text_test = re.sub(r',,+', ',', text_test)
    # text_test = re.sub(r',', ' ', text_test)
    # x_test = open(test_filename + '.num', "w")
    # x_test.writelines(text_test)
    # x_test.close()

    break
