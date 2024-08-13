import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import export_graphviz
import graphviz

base = pd.read_csv('insurance.csv')
base = base.drop(['Unnamed: 0'], axis=1)
print(base)

base = base.fillna(method='ffill')

y = base.iloc[:,7].values
x = base.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values

Labelencoder = LabelEncoder()

for i in range(x.shape[1]):
    if x[:,i].dtype == 'object':
        x[:,i] = Labelencoder.fit_transform(x[:,i])

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y,test_size=0.3, random_state=1)

modelo =  DecisionTreeClassifier(random_state=1)
modelo.fit(x_treinamento, y_treinamento)

previsoes = modelo.predict(x_teste)

accuracy = accuracy_score(y_teste, previsoes)
precision = precision_score(y_teste, previsoes, average='weighted')
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')
print(f'Acuracia: {accuracy}, Precisão: {precision}, Recall: {recall}, F1: {f1}')