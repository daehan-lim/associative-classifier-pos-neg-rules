import urllib.request
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

records = []
# directly load from the url instead of using the file
for line in urllib.request.urlopen("https://user.informatik.uni-goettingen.de/~sherbold/store_data.csv"):
    # this also means we need to decode the binary string into ascii
    records.append(line.decode('ascii').strip().split(','))
print('Records:')
print(records)

# we first need to create a one-hot encoding of our transactions
te = TransactionEncoder()
te_ary = te.fit_transform(records)
data_df = pd.DataFrame(te_ary, columns=te.columns_)

# use support of 0.005 - low threshold may include to many candidates
# careful selection of rules based on other metrics required
# this means that we use a higher confidence
frequent_itemsets = apriori(pd.DataFrame(data_df), min_support=0.005, use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence",
                  min_threshold=0.5).sort_values('lift', ascending=False)
print(rules)
