import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2, SelectKBest

anuncio = pd.read_csv('ad.data', header=None)
#print(anuncio.shape)

x = anuncio.iloc[:,:-1].values
y = anuncio.iloc[:, -1].values

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(x,y, test_size=0.3, random_state=0)

modelo1 = GaussianNB()
modelo1.fit(X_treinamento,y_treinamento)
previsao1 = modelo1.predict(X_teste)
print(accuracy_score(y_teste, previsao1))

selecao = SelectKBest(chi2, k=7)
X_novo = selecao.fit_transform(x,y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X_novo, y, test_size=0.3, random_state=0)

modelo2 = GaussianNB()
modelo2.fit(X_treinamento, y_treinamento)
previsao2 = modelo2.predict(X_teste)
print(accuracy_score(y_teste, previsao2))