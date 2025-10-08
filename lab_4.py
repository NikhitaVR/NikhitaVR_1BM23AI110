#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
data = {('USA', 2020): [331000000, 21000000, 78.5],
        ('USA', 2021): [333000000, 21500000, 78.7],
        ('China', 2020): [1402000000, 14700000, 76.9],
        ('China', 2021): [1405000000, 15500000, 77.1],
        ('Germany', 2020): [83000000, 4000000, 81.5],
        ('Germany', 2021): [83100000, 4200000, 81.5]}
index = pd.MultiIndex.from_tuples(data.keys(), names=['Country', 'year'])
df = pd.DataFrame(list(data.values()), index = index, columns= ['Population', 'GDP', 'Life'])
print("Original dataframe")
print(df, "\n")
print("summary statistics using pandas: ")
print(df[['Population', 'GDP']].agg(['mean', 'sum', 'max', 'min', 'std']), "\n")
print("Summary Statistics using numpy: ")
print("Mean (Population):", np.mean(df['Population']))
print("Sum (GDP):", np.sum(df['GDP']))
print("Max (Population):", np.max(df['Population']))
print("Min(GDP): ", np.min(df['GDP']), "\n")
df['GDP'] = df['GDP'].apply(lambda x: x*1.10)
print("After increasing GDP by 10% : ")
print(df, "\n")
swapped_df = df.swaplevel().sort_index(level = 'year')
print("After swaplevel() and sorting by year:")
print(swapped_df, "\n")
pivot_df = df.unstack(level='year')
print("After unstack() - Pivoted View : ")
print(pivot_df, "\n")
print("Population trends for each country: ")
for country in df.index.get_level_values('Country').unique():
    print(f"\n{country}:")
    print(df.loc[country]['Population'])


# In[ ]:





# In[ ]:




