import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
#from sklearn.tree import export_graphviz
#import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


base = pd.read_csv('insurance.csv')

base = base.drop(columns=['Unnamed: 0'], axis=1)

print("Valore nulos:\n", base.isnull().sum())

for column in base.columns:
    if base[column].dtype == 'object':
        base[column] = base[column].fillna(base[column].mode()[0])
    else:
        base[column] = base[column].fillna(base[column].mean())

print("Valore nulos:\n", base.isnull().sum())

y = base.iloc[:,7].values
X = base.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values

labelencoder = LabelEncoder()

for i in range(X.shape[1]):
    if X[:, i].dtype == 'object':
        X[:, i] = labelencoder.fit_transform(X[:, i])

X_treinamento, x_teste, y_treinamento, y_teste = train_test_split(X,y, test_size=0.3, random_state=1)

modelo = RandomForestClassifier(random_state=1, n_estimators=500)
modelo = RandomForestClassifier(random_state=1, n_estimators=500, max_depth=5, max_leaf_nodes=9)
modelo.fit(X_treinamento, y_treinamento)

tree_index = 0
tree_to_visualize = modelo.estimators_[tree_index]
plt.figure(figsize=(20,10))
plot_tree(tree_to_visualize, filled=True, feature_names=base.columns[:-1], class_names=True, rounded=True)
plt.show()

previsao = modelo.predict(x_teste)

accuracy = accuracy_score(y_teste, previsao)
precision = precision_score(y_teste, previsao, average='weighted')
recall = recall_score(y_teste, previsao, average='weighted')
f1 = f1_score(y_teste, previsao, average='weighted')

print(f'Acuracia: {accuracy}, Precisão: {precision}, Recall: {recall}, F1: {f1}')

report = classification_report(y_teste, previsao)
print(report)