import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('cars.xls')

print(df.head())

scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())

print(X)

est = sm.OLS(y, X).fit()

print(est.summary())

print(y.groupby(df.Doors).mean())
