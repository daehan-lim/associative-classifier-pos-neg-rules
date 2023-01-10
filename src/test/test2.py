import pandas as pd

# rule_antecedents = [{'diaper', 'kid stuff'}]
# rule_consequents = [{'beer'}]
# df = pd.DataFrame({'antecedents': rule_antecedents, 'consequents': rule_consequents})
# df

# create a DataFrame with two columns: 'antecedents' and 'consequents'
df = pd.DataFrame(columns=['antecedents', 'consequents'])

# add a new row to the DataFrame
df.loc[0] = {'antecedents': {'diaper', 'kid stuff'}, 'consequents': {'beer'}}

print(df)