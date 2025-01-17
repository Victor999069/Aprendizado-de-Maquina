import pandas as pd
import numbers as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

base = pd.read_csv('mt_cars.csv')
base = base.drop(['Unnamed: 0'], axis=1)
#print(base.head())

# corr = base.corr()
# sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
# plt.show()

# colunas = [('mpg', 'hp'), ('mpg', 'drat'), ('mpg', 'cyl'), ('mpg', 'wt')]
# plots = len(colunas)

# fig, axes = plt.subplots(nrows=plots, ncols=1, figsize=(4,4 * plots))

# for i, pair in enumerate(colunas):
#     x_col, y_col = pair
#     sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
#     axes[i].set_title(f'{x_col} vs {y_col}')

# plt.tight_layout()
# plt.show()

#------------------------------------------------------------------------------

#AIC: 156.6
#BIC: 162.5
modelo = sm.ols(formula='mpg ~ wt + disp + hp', data=base)
modelo = modelo.fit()
modelo.summary()
#print(modelo.summary())

residuos = modelo.resid
# plt.hist(residuos, bins=20)
# plt.xlabel('Residuos')
# plt.ylabel('Frequencia')
# plt.title('Histograma')
#plt.show()

stats.probplot(residuos, dist='norm', plot=plt)
plt.title('Q-Q Plot Residuo')
plt.show()