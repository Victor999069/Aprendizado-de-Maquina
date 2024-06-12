import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Carregar o conjunto de dados
df = pd.read_csv('melb_data.csv')

iris = datasets.load_iris()
X = iris.data[:, 2:4]  # Usar "petal length" e "petal width"
y = iris.target

X = X[y != 2]
y = y[y != 2]

X = np.vstack([X, [3, 1.5]])
y = np.append(y, 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def plot_svm_decision_boundary(model, X, y, title):
    h = .02  # Tamanho do passo na malha
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Petal length (standardized)')
    plt.ylabel('Petal width (standardized)')
    plt.title(title)
    plt.show()

svm_hard = SVC(kernel='linear', C=1e10)
svm_hard.fit(X_scaled, y)
plot_svm_decision_boundary(svm_hard, X_scaled, y, 'SVM de Margens Rígidas')

C_values = [1, 10, 100, 1000]
for C in C_values:
    svm_soft = SVC(kernel='linear', C=C)
    svm_soft.fit(X_scaled, y)
    plot_svm_decision_boundary(svm_soft, X_scaled, y, f'SVM de Margens Suaves (C={C})')

svm_poly = SVC(kernel='poly', degree=3, C=1)
svm_poly.fit(X_scaled, y)
plot_svm_decision_boundary(svm_poly, X_scaled, y, 'SVM com Kernel Polinomial')

svm_rbf = SVC(kernel='rbf', gamma=1, C=1)
svm_rbf.fit(X_scaled, y)
plot_svm_decision_boundary(svm_rbf, X_scaled, y, 'SVM com Kernel Gaussiano')

gamma_values = [0.1, 1, 10]
C_values = [1, 10, 100, 1000]
for gamma in gamma_values:
    for C in C_values:
        svm_rbf = SVC(kernel='rbf', gamma=gamma, C=C)
        svm_rbf.fit(X_scaled, y)
        plot_svm_decision_boundary(svm_rbf, X_scaled, y, f'SVM com Kernel Gaussiano (gamma={gamma}, C={C})')
