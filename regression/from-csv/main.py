import numpy as np
import pandas as pd
from sklearn import model_selection, metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#region Loading data
def loadData():
  df = pd.read_csv('sales-data.csv')

  # Converter a coluna de data para o tipo datetime
  df['data'] = pd.to_datetime(df['data'])
  df['mes'] = df['data'].dt.month
  df = df[df['mes'].isin([3,4])]

  # Agrupar os dados por 'data' e somar as quantidades vendidas
  df_grouped = df.groupby('data')['quantidade'].sum().reset_index()

  # Ordenar os dados pela data e resetar o índice para criar um índice sequencial
  df_grouped = df_grouped.sort_values(by='data').reset_index(drop=True)
  df_grouped['indice'] = df_grouped.index + 1  # Criar um índice sequencial começando em 1

  easterDays = [
    '2011-04-24',
    '2012-04-8',
    '2013-03-31',
    '2014-04-20',
    '2015-04-05',
    '2016-03-27',
    '2017-04-16',
    '2018-04-01',
    '2019-04-21',
    '2020-04-12',
    '2021-04-04',
    '2022-04-17',
    '2023-04-09',
    '2024-03-31',
              ]
  easterDays = pd.to_datetime(easterDays)
  df_grouped['is_holiday'] = 0

  for pascoa in easterDays:
      pascoa_index = df_grouped.index[df_grouped['data'] == pascoa].tolist()
      if not pascoa_index:
          # Se a data da Páscoa não for encontrada, pegar o próximo dia disponível
          pascoa_index = df_grouped.index[df_grouped['data'] > pascoa].tolist()
      
      if pascoa_index:
          pascoa_index = pascoa_index[0]
          for i in range(11):
              if pascoa_index - i >= 0:
                  peso = np.exp(-0.2 * i)  # Função exponencial decrescente
                  df_grouped.loc[pascoa_index - i, 'is_holiday'] = peso
  
  # Separar as variáveis independentes (X) e dependentes (y)
  X = df_grouped[['indice', 'is_holiday']].values
  y = df_grouped['quantidade'].values.reshape(-1, 1)
  return (X, y, df_grouped)

#endregion

#region utils
def remove_prefix(params):
  """Removes the "knn__" prefix from keys in a dictionary.

  Args:
      params: A dictionary containing KNeighborsRegressor parameters.

  Returns:
      A new dictionary with the prefix removed from keys.
  """
  new_params = {}
  for key, value in params.items():
    new_key = key.split("__")[1]  # Split and remove the first part (knn__)
    new_params[new_key] = value
  return new_params
#endregion

X, y, dataFrame = loadData()

#region Cross validation
innerKfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=False)
outerKFolds = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=False)

#region parameters tuning
knn_pipe = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ('knn', KNeighborsRegressor()),
])
knnGridSearch = model_selection.GridSearchCV(
    estimator=knn_pipe,
    param_grid={'knn__n_neighbors': list(np.arange(1, 15)), 'knn__weights':['uniform', 'distance'], 'knn__metric': ['manhattan', 'euclidean', 'minkowski']},
    scoring='neg_mean_squared_error',
    cv=outerKFolds,
    refit=True
)
knnGridSearch.fit(X, y)
knnBestParams = knnGridSearch.best_params_
print(f"KNN best params: {knnBestParams}")

linear_pipe = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ('linear', LinearRegression()),
])
linearGridSearch = model_selection.GridSearchCV(
  estimator=linear_pipe,
  cv=outerKFolds,
  param_grid={},
  scoring='neg_mean_squared_error', 
  refit=True)
linearGridSearch.fit(X, y)
linearBestParams = linearGridSearch.best_params_
print(f"Linear best params: {linearBestParams}")

mlp_pipe = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ('mlp', MLPRegressor()),
])
mlpGridSearch = model_selection.GridSearchCV(
  estimator=mlp_pipe,
  param_grid={'mlp__max_iter': [500], 'mlp__activation': ['logistic'], 'mlp__hidden_layer_sizes':[(25, 2), (25, 3), (50, 2), (50, 3), (100, 2), (100, 3)], 
              'mlp__solver': ['sgd'], 'mlp__alpha': [0.01, 0.001, 0.0001], 'mlp__tol': [1e-9]}, 
  cv=outerKFolds, 
  scoring='neg_mean_squared_error', 
  refit=True)
mlpGridSearch.fit(X, y.ravel())
mlpBestParams = mlpGridSearch.best_params_
print(f"MLP best params: {mlpBestParams}")


poly_pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression()),
])
polyGridSearch = model_selection.GridSearchCV(
    estimator=poly_pipe,
    param_grid={'poly__degree': np.arange(1, 15)},
    scoring='neg_mean_squared_error',
    cv=outerKFolds,
    refit=True
)
polyGridSearch.fit(X, y)
polyBestParams = polyGridSearch.best_params_
print(f"Poly best params: {polyBestParams}")

svr_pipe = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ('svr', SVR()),
])
svrGridSearch = model_selection.GridSearchCV(
    estimator=svr_pipe,
    param_grid={'svr__kernel': ['linear', 'sigmoid', 'rbf'], 'svr__C': [0.3, 0.5, 0.7, 1]},
    scoring='neg_mean_squared_error',
    cv=outerKFolds,
    refit=True
)
svrGridSearch.fit(X, y.ravel())
svrBestParams = svrGridSearch.best_params_
print(f"svr best params: {svrBestParams}")


knnErrors = []
linearRegressionErrors = []
mlpErrors = []
polyErrors = []
svrErrors = []

normalizer = StandardScaler()
for trainIndexes, testIndexes in innerKfold.split(X, y):
  X_train, y_train = X[trainIndexes], y[trainIndexes]
  normalizer.fit(X_train)
  X_train = normalizer.transform(X_train)
  X_test, y_test = X[testIndexes], y[testIndexes]
  X_test = normalizer.transform(X_test)

  knn = KNeighborsRegressor(**remove_prefix(knnBestParams))
  knn.fit(X_train, y_train)
  knn_predictions = knn.predict(X_test)
  knnError = metrics.mean_squared_error(y_test, knn_predictions)
  knnErrors.append(knnError)

  linearRegression = LinearRegression(**remove_prefix(linearBestParams))
  linearRegression.fit(X_train, y_train)
  linear_predictions = linearRegression.predict(X_test)
  linearError = metrics.mean_squared_error(y_test, linear_predictions)
  linearRegressionErrors.append(linearError)
  
  mlp = MLPRegressor(**remove_prefix(mlpBestParams))
  mlp.fit(X_train, y_train.ravel())
  mlp_predictions = mlp.predict(X_test)
  mlpError = metrics.mean_squared_error(y_test, mlp_predictions)
  mlpErrors.append(mlpError)

  polyModel = LinearRegression()
  polyFeat = PolynomialFeatures(**remove_prefix(polyBestParams), include_bias=False)
  poly_features = polyFeat.fit_transform(X_train)
  polyModel.fit(poly_features, y_train)
  poly_predictions = polyModel.predict(polyFeat.fit_transform(X_test))
  polyError = metrics.mean_squared_error(y_test, poly_predictions)
  polyErrors.append(polyError)

  svr = SVR(**remove_prefix(svrBestParams))
  svr.fit(X_train, y_train.ravel())
  svr_predictions = svr.predict(X_test)
  svrError = metrics.mean_squared_error(y_test, svr_predictions)
  svrErrors.append(svrError)

print(f"KNN - media dos mse: \n {np.mean(knnErrors)}")
print(f"Linear regression - media dos mse: \n {np.mean(linearRegressionErrors)}")
print(f"MLP - media dos mse: \n {np.mean(mlpErrors)}")
print(f"Poly - media dos mse: \n {np.mean(polyErrors)}")
print(f"SVR - media dos mse: \n {np.mean(svrErrors)}")

knn_predictions = knnGridSearch.predict(X)
dataFrame['knn_predictions'] = knn_predictions

linear_predictions = linearGridSearch.predict(X)
dataFrame['linear_predictions'] = linear_predictions

mlp_predictions = mlpGridSearch.predict(X)
dataFrame['mlp_predictions'] = mlp_predictions

poly_predictions = polyGridSearch.predict(X)
dataFrame['poly_predictions'] = poly_predictions

svr_predictions = svrGridSearch.predict(X)
dataFrame['svr_predictions'] = svr_predictions

#endregion
#endregion

# Plotar os dados e as previsões
plt.scatter(dataFrame['indice'], y, color='blue', label='Dados Reais')
plt.plot(dataFrame['indice'], dataFrame['knn_predictions'], color='gray', label='Previsões KNN')
plt.plot(dataFrame['indice'], dataFrame['linear_predictions'], color='green', label='Previsões Linear')
plt.plot(dataFrame['indice'], dataFrame['mlp_predictions'], color='yellow', label='Previsões MLP')
plt.plot(dataFrame['indice'], dataFrame['poly_predictions'], color='orange', label='Previsões Poly')
plt.plot(dataFrame['indice'], dataFrame['svr_predictions'], color='purple', label='Previsões SVR')

ticks = []
ticksLabels = []
for index, date in zip(dataFrame['indice'], dataFrame['data']):
  if date.day == 1:
    ticks.append(index)
    ticksLabels.append(f"{date.day}/{date.month}/{date.year}")
plt.xticks(ticks=ticks, labels=ticksLabels, rotation=45)
plt.ylabel('Quantidade Vendida')
plt.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()
#endregion