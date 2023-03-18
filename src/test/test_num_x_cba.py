import random
import pandas as pd
import csv
import numpy as np
from util import util
from pathlib import Path

with open('../../data/dataset.csv', 'r') as file:
    dataset = [list(filter(None, row)) for row in csv.reader(file)]

transactions_df = util.convert_trans_to_df(dataset)
random.seed(0)
transactions_0 = pd.DataFrame(transactions_df[transactions_df['0']].reset_index(drop=True))
transactions_1 = pd.DataFrame(
    transactions_df[transactions_df['1']].reset_index(drop=True))

indices = list(range(0, len(transactions_0)))
random.shuffle(indices)
test_set_0 = transactions_0.iloc[indices[:417], :].reset_index(drop=True)
training_set_0 = transactions_0.iloc[indices[417:], :].reset_index(drop=True)

indices = list(range(0, len(transactions_1)))
random.shuffle(indices)
test_set_1 = transactions_1.iloc[indices[:43], :].reset_index(drop=True)
training_set_1 = transactions_1.iloc[indices[43:], :].reset_index(drop=True)

training_set = pd.concat([training_set_0, training_set_1])
test_set = pd.concat([test_set_0, test_set_1])

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

filepath = Path('../../data/training_cba.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
training_transactions.to_csv(filepath)

filepath = Path('../../data/test_cba.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
test_transactions.to_csv(filepath)
