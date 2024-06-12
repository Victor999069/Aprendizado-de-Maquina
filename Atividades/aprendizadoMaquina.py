import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('melb_data.csv')

print(df.head())

print(df.isnull().sum())

df['Car'] = df['Car'].fillna(df['Car'].mode()[0])
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].mean())
df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())
df['CouncilArea'] = df['CouncilArea'].fillna(df['CouncilArea'].mode()[0])

df['Price'] = df['Price'].fillna(df['Price'].mean())

df = pd.get_dummies(df, columns=['Suburb', 'Type', 'CouncilArea', 'Regionname', 'Method', 'SellerG'], drop_first=True)

non_numeric = df.select_dtypes(exclude=[np.number]).columns
if len(non_numeric) > 0:
    print("Ainda existe colunas com ZERO", non_numeric)
else:
    print("Não existe")

X = df.drop(['Price', 'Address', 'Date'], axis=1) 
Y = df['Price']

X_train, X_test, Y_Train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_Train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Comparação entre Valores Reais e Previstos')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red')
plt.show()

residuals = Y_test - Y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Resíduos')
plt.title('Distribuição dos Resíduos')
plt.show()