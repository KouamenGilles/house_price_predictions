# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:07:19 2023

@author: Moi
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None
df = pd.read_csv("housing.csv", sep=",")

df.head(10)
df.info()
df.describe()
df = df.astype({'ocean_proximity':'category'})

df['rooms_per_household'] = df['total_rooms']/df['households']
df['rooms_per_household']

df['bedrooms_per_household'] = df['total_bedrooms']/df['households']
df['population_per_household'] = df['population']/df['households']
df = df.drop(['latitude','longitude'], axis=1)
df.info()

df.describe()

df.hist(bins=50, figsize=(20,15))
plt.show()

df['ocean_proximity'].value_counts().plot(kind='bar')
plt.show()

corr_matrix = df.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
corr_matrix['total_rooms'].sort_values(ascending=False)


cmap = sns.diverging_palette(220,10, as_cmap=True)
sns.heatmap(corr_matrix, cmap=cmap,cbar_kws={"shrink":.5}, linewidths=.5)


## preparation des données

df.describe()['median_house_value']
df['median_house_value'].hist(bins=50)
plt.xlabel('Valeur mediane des maison (en dollars')
plt.ylabel('Féquence')
plt.title('Distribution des prix des maison en californie')
plt.show()


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.1, random_state=42) 

price_train = train_set['median_house_value']
train_set = train_set.drop(['median_house_value'], axis=1)

price_test = test_set['median_house_value']
test_set = test_set.drop(['median_house_value'], axis=1)

df.columns
list(df)
df.index
df.values
type(df.values)

train_set.loc[1,:]
