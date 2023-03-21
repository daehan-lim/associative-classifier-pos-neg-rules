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
    training_set = transactions_df.iloc[indices[460:], :].reset_index(drop=True)
    test_set = transactions_df.iloc[indices[:460], :].reset_index(drop=True)

    arr = np.arange(training_set.values.shape[1])
    new_column_names = {col: str(i - 1) for i, col in enumerate(training_set.columns)}
    new_column_names['0'] = 148
    new_column_names['1'] = 149

    training_set.rename(columns=new_column_names, inplace=True)
    test_set.rename(columns=new_column_names, inplace=True)

    training_transactions = training_set.drop([148, 149], axis=1).apply(lambda row: row.index[row], axis=1).tolist()
    training_transactions = pd.DataFrame(training_transactions)
    training_transactions.fillna("", inplace=True)

    y_training = training_set.apply(lambda row: 148 if row[148] else 149, axis=1).tolist()
    training_transactions['class'] = y_training

    test_transactions = test_set.drop([148, 149], axis=1).apply(lambda row: row.index[row], axis=1).tolist()
    test_transactions = pd.DataFrame(test_transactions)
    test_transactions.fillna("", inplace=True)

    y_test = test_set.apply(lambda row: 148 if row[148] else 149, axis=1).tolist()
    test_transactions['class'] = y_test

    training_filename = '../../data/training_cba' + str(seed)
    filepath = Path(training_filename + '.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    training_transactions.to_csv(filepath, index=False, header=False)

    test_filename = '../../data/test_cba' + str(seed)
    filepath = Path(test_filename + '.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    test_transactions.to_csv(filepath, index=False, header=False)

    training_filename = '../../data/training_cba' + str(seed)
    text = open(training_filename + '.csv', "r")
    text = ''.join([i for i in text])
    text = re.sub(r',,+', ',', text)
    text = re.sub(r',', ' ', text)
    x = open(training_filename + '.num', "w")
    x.writelines(text)
    x.close()

    test_filename = '../../data/test_cba' + str(seed)
    text_test = open(test_filename + '.csv', "r")
    text_test = ''.join([i for i in text_test])
    text_test = re.sub(r',,+', ',', text_test)
    text_test = re.sub(r',', ' ', text_test)
    x_test = open(test_filename + '.num', "w")
    x_test.writelines(text_test)
    x_test.close()

    break
