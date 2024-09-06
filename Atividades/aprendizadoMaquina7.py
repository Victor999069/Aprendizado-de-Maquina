from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from skmultilearn.adapt import MLARAM
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
import pandas as pd

musica = pd.read_csv('Musica.csv')

classe = musica.iloc[:,0:6].values
previsores = musica.iloc[:,7:78].values

X_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size=0.3, random_state=0)

mlran = MLARAM()
mlran.fit(X_treinamento, y_treinamento)
previsto = mlran.predict(x_teste)
print(hamming_loss(y_teste, previsto))


binary = BinaryRelevance(classifier=SVC())
binary.fit(X_treinamento, y_treinamento)
previsto = binary.predict(x_teste)
print(hamming_loss(y_teste, previsto))

classife = ClassifierChain(classifier=SVC())
classife.fit(X_treinamento, y_treinamento)
previsto = classife.predict(x_teste)
print(hamming_loss(y_teste, previsto))

label = LabelPowerset(classifier=SVC())
label.fit(X_treinamento, y_treinamento)
previsto = label.predict(x_teste)
print(hamming_loss(y_teste, previsto))