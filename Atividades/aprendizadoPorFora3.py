import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import export_graphviz
import graphviz

base = pd.read_csv('insurance.csv')

base = base.drop(columns = ['Unnamed: 0'], axis=1)

print("Valore nulos:\n", base.isnull().sum())

for column in base.columns:
    if base[column].dtype == 'object':
        base[column] = base[column].fillna(base[column].mode()[0])
    else:
        base[column] = base[column].fillna(base[column].mean())

print("Valore nulos:\n", base.isnull().sum())

y = base.iloc[:,7].values
X = base.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values

labelEncoder = LabelEncoder()

for i in range(X.shape[1]):
    if X[:, i].dtype == 'object':
        X[:, i] = labelEncoder.fit_transform(X[:, i])

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3, random_state=1)

#modelo = DecisionTreeClassifier(random_state=1)
modelo = DecisionTreeClassifier(random_state=1, max_depth=8)
modelo.fit(X_treinamento, y_treinamento)

#dot_data = export_graphviz(modelo, out_file=None, filled=True, feature_names=base.columns[: -1], class_names=True, rounded=True)
#graph = graphviz.Source(dot_data)
#graph.render("decision_tree", format="png")
#graph.view()


previsoes = modelo.predict(X_teste)

accurary = accuracy_score(y_teste, previsoes)
precision = precision_score(y_teste, previsoes, average='weighted')
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')

print(f'Acuracia: {accurary}, Precisão: {precision}, Recall: {recall}, F1: {f1}')

report = classification_report(y_teste, previsoes)

print(report)

