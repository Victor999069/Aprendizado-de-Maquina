import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC

credito = pd.read_csv('Credit_simple.csv', sep=';')

count = credito.groupby(['CLASSE']).size()

y = credito['CLASSE'].values
x = credito.iloc[:,:-1].values

label = LabelEncoder()

for i in range(x.shape[1]):
    if x[:, i].dtype == 'object':
        x[:, i] = label.fit_transform(x[:, i])

sm = SMOTENC(random_state=0, categorical_features=[3,5,6])
X_res, y_res = sm.fit_resample(x, y)

df = pd.DataFrame({'CLASSE': y_res})
df.value_counts()

print(count)

print(df.value_counts())