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

def week_of_month(date_value):
  week = date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1
  return date_value.isocalendar()[1] if week < 0 else week
#endregion

#region Loading data
def loadData():
  df = pd.read_csv('sales-data.csv')

  # Converter a coluna de data para o tipo datetime
  df['data'] = pd.to_datetime(df['data'])
  df['mes'] = df['data'].dt.month
  df = df[~((df['mes'] == 5) & (df['data'].dt.year == 2024))]
  df = df[df['mes'].isin([3,4])]


  # Agrupar os dados por 'data' e somar as quantidades vendidas
  df_grouped = df.groupby('data')['quantidade'].sum().reset_index()

  # Ordenar os dados pela data e resetar o índice para criar um índice sequencial
  df_grouped = df_grouped.sort_values(by='data').reset_index(drop=True)
  df_grouped['indice'] = df_grouped.index + 1  # Criar um índice sequencial começando em 1

  for i in range(0,7):
        df_grouped[f'weekday_{i}'] = df_grouped['data'].apply(lambda x: 1 if x.weekday() == i else 0)

  for i in range(1, 6):
      df_grouped[f'week_of_month_{i}'] = df_grouped['data'].apply(lambda x: 1 if week_of_month(x) == i else 0)
      
 # Adicionar colunas de vendas dos dias anteriores
  for i in range(1, 6):
      df_grouped[f'previous_day_sales_{i}'] = df_grouped['quantidade'].shift(i)

  # adicionando feature easter_influence
  df_grouped['easter_influence'] = 0
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

  pascoaIndexes = []
  for pascoa in easterDays:
      pascoa_index = df_grouped.index[df_grouped['data'] == pascoa].tolist()
      if not pascoa_index:
          # Se a data da Páscoa não for encontrada, pegar o próximo dia disponível
          pascoa = pascoa - pd.Timedelta(days=1)
          pascoa_index = df_grouped.index[df_grouped['data'] == pascoa].tolist()
      
      if pascoa_index:
          pascoa_index = pascoa_index[0]
          for i in range(15):
              if pascoa_index - i >= 0:
                  peso = np.exp(-0.2 * i)  # Função exponencial decrescente
                  df_grouped.loc[pascoa_index - i, 'easter_influence'] = peso
      pascoaIndexes.append(pascoa_index)
  # Separar as variáveis independentes (X) e dependentes (y)
  X = df_grouped[['easter_influence', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'week_of_month_1', 'week_of_month_2', 'week_of_month_3', 'week_of_month_4', 'week_of_month_5']].values
  # X = df_grouped[['indice', 'easter_influence', 'week_of_month_1', 'week_of_month_2', 'week_of_month_3', 'week_of_month_4', 'week_of_month_5']].values
  y = df_grouped['quantidade'].values.reshape(-1,  1)
  return (X, y, df_grouped, pascoaIndexes)

#endregion

X, y, dataFrame, pascoaIndexes = loadData()
dataFrame.to_csv('newdata.csv')
#region Cross validation
innerKfold = model_selection.KFold(n_splits=10, shuffle=False)
outerKFolds = model_selection.KFold(n_splits=3, shuffle=False)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.16, shuffle=False)

#region parameters tuning
knn_pipe = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ('knn', KNeighborsRegressor()),
])
knnGridSearch = model_selection.GridSearchCV(
    estimator=knn_pipe,
    param_grid={'knn__n_neighbors': list(np.arange(1, 15)), 'knn__weights':['uniform', 'distance'], 'knn__metric': ['manhattan', 'euclidean', 'minkowski']},
    scoring='neg_mean_absolute_error',
    cv=outerKFolds,
    refit=True
)
knnGridSearch.fit(X_train, y_train)
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
  scoring='neg_mean_absolute_error', 
  refit=True)
linearGridSearch.fit(X_train, y_train)
linearBestParams = linearGridSearch.best_params_
print(f"Linear best params: {linearBestParams}")

mlp_pipe = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ('mlp', MLPRegressor()),
])
mlpGridSearch = model_selection.GridSearchCV(
  estimator=mlp_pipe,
  param_grid={'mlp__max_iter': [10000], 'mlp__activation': ['relu'], 'mlp__hidden_layer_sizes':[(25, 2), (25, 3), (25, 4), (50, 2), (50, 3), (50, 4), (100, 2), (100, 3), (100, 4)], 
              'mlp__solver': ['sgd'], 'mlp__alpha': [0.0001], 'mlp__tol': [1e-9], 'mlp__learning_rate_init': [0.1, 0.01 ,0.001]}, 
  cv=outerKFolds, 
  scoring='neg_mean_absolute_error', 
  refit=True)
mlpGridSearch.fit(X_train, y_train.ravel())
mlpBestParams = mlpGridSearch.best_params_
print(f"MLP best params: {mlpBestParams}")


poly_pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression()),
])
polyGridSearch = model_selection.GridSearchCV(
    estimator=poly_pipe,
    param_grid={'poly__degree': np.arange(1, 5)},
    scoring='neg_mean_absolute_error',
    cv=outerKFolds,
    refit=True
)
polyGridSearch.fit(X_train, y_train)
polyBestParams = polyGridSearch.best_params_
print(f"Poly best params: {polyBestParams}")

svr_pipe = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ('svr', SVR()),
])
svrGridSearch = model_selection.GridSearchCV(
    estimator=svr_pipe,
    param_grid={'svr__kernel': ['linear', 'sigmoid', 'rbf'], 'svr__C': [0.3, 0.5, 0.7, 1]},
    scoring='neg_mean_absolute_error',
    cv=outerKFolds,
    refit=True
)
svrGridSearch.fit(X_train, y_train.ravel())
svrBestParams = svrGridSearch.best_params_
print(f"svr best params: {svrBestParams}")


knnErrors = []
linearRegressionErrors = []
mlpErrors = []
polyErrors = []
svrErrors = []

for trainIndexes, testIndexes in innerKfold.split(X, y):
  normalizer = StandardScaler()
  X_train_internal, y_train_internal = X[trainIndexes], y[trainIndexes]
  normalizer.fit(X_train_internal)
  X_train_internal = normalizer.transform(X_train_internal)
  X_test_internal, y_test_internal = X[testIndexes], y[testIndexes]
  X_test_internal = normalizer.transform(X_test_internal)

  knn = KNeighborsRegressor(**remove_prefix(knnBestParams))
  knn.fit(X_train_internal, y_train_internal)
  knn_predictions = knn.predict(X_test_internal)
  knnError = metrics.mean_absolute_error(y_test_internal, knn_predictions)
  knnErrors.append(knnError)

  linearRegression = LinearRegression(**remove_prefix(linearBestParams))
  linearRegression.fit(X_train_internal, y_train_internal)
  linear_predictions = linearRegression.predict(X_test_internal)
  linearError = metrics.mean_absolute_error(y_test_internal, linear_predictions)
  linearRegressionErrors.append(linearError)
  
  mlp = MLPRegressor(**remove_prefix(mlpBestParams))
  mlp.fit(X_train_internal, y_train_internal.ravel())
  mlp_predictions = mlp.predict(X_test_internal)
  mlpError = metrics.mean_absolute_error(y_test_internal, mlp_predictions)
  mlpErrors.append(mlpError)

  polyModel = LinearRegression()
  polyFeat = PolynomialFeatures(**remove_prefix(polyBestParams), include_bias=False)
  poly_features = polyFeat.fit_transform(X_train_internal)
  polyModel.fit(poly_features, y_train_internal)
  poly_predictions = polyModel.predict(polyFeat.fit_transform(X_test_internal))
  polyError = metrics.mean_absolute_error(y_test_internal, poly_predictions)
  polyErrors.append(polyError)

  svr = SVR(**remove_prefix(svrBestParams))
  svr.fit(X_train_internal, y_train_internal.ravel())
  svr_predictions = svr.predict(X_test_internal)
  svrError = metrics.mean_absolute_error(y_test_internal, svr_predictions)
  svrErrors.append(svrError)

print(f"KNN - media dos mae: \n {np.mean(knnErrors)}")
print(f"Linear regression - media dos mae: \n {np.mean(linearRegressionErrors)}")
print(f"MLP - media dos mae: \n {np.mean(mlpErrors)}")
print(f"Poly - media dos mae: \n {np.mean(polyErrors)}")
print(f"SVR - media dos mae: \n {np.mean(svrErrors)}")

resultsDF = pd.DataFrame(y_test)

knn_predictions = knnGridSearch.predict(X_test)
resultsDF['knn_predictions'] = knn_predictions

linear_predictions = linearGridSearch.predict(X_test)
resultsDF['linear_predictions'] = linear_predictions

mlp_predictions = mlpGridSearch.predict(X_test)
resultsDF['mlp_predictions'] = mlp_predictions

poly_predictions = polyGridSearch.predict(X_test)
resultsDF['poly_predictions'] = poly_predictions

svr_predictions = svrGridSearch.predict(X_test)
resultsDF['svr_predictions'] = svr_predictions

#endregion
#endregion

# Plotar os dados e as previsões

indices = range(len(X_test))
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.tight_layout(pad=5.0)

y_min = min(y_test.min(), resultsDF.min().min())
y_max = max(y_test.max(), resultsDF.max().max())

# Dados Reais
axs[0, 0].scatter(indices, y_test, color='blue', label='Real data', marker="o", alpha=0.3)
axs[0, 0].set_title('Real data')
axs[0, 0].set_ylabel('Ammount sold')
axs[0, 0].set_ylim(y_min, y_max)
axs[0, 0].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[0, 0].legend()
axs[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões KNN
axs[0, 1].scatter(indices, resultsDF['knn_predictions'], color='gray', label='Predicted KNN', marker="o", alpha=0.3)
axs[0, 1].set_title('Predicted KNN')
axs[0, 1].set_ylabel('Ammount sold')
axs[0, 1].set_ylim(y_min, y_max)
axs[0, 1].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[0, 1].legend()
axs[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões Linear
axs[1, 0].scatter(indices, resultsDF['linear_predictions'], color='green', label='Predicted Linear', marker="o", alpha=0.3)
axs[1, 0].set_title('Predicted Linear')
axs[1, 0].set_ylabel('Ammount sold')
axs[1, 0].set_ylim(y_min, y_max)
axs[1, 0].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[1, 0].legend()
axs[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões MLP
axs[1, 1].scatter(indices, resultsDF['mlp_predictions'], color='yellow', label='Predicted MLP', marker="o", alpha=0.3)
axs[1, 1].set_title('Predicted MLP')
axs[1, 1].set_ylabel('Ammount sold')
axs[1, 1].set_ylim(y_min, y_max)
axs[1, 1].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[1, 1].legend()
axs[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões Poly
axs[2, 0].scatter(indices, resultsDF['poly_predictions'], color='orange', label='Predicted Poly', marker="o", alpha=0.3)
axs[2, 0].set_title('Predicted Poly')
axs[2, 0].set_ylabel('Ammount sold')
axs[2, 0].set_ylim(y_min, y_max)
axs[2, 0].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[2, 0].legend()
axs[2, 0].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões SVR
axs[2, 1].scatter(indices, resultsDF['svr_predictions'], color='purple', label='Predicted SVR', marker="o", alpha=0.3)
axs[2, 1].set_title('Predicted SVR')
axs[2, 1].set_ylabel('Ammount sold')
axs[2, 1].set_ylim(y_min, y_max)
axs[2, 1].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[2, 1].legend()
axs[2, 1].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Adicionando ticks e labels
ticks = []
ticksLabels = []
for index, date, dataIndex in zip(indices, dataFrame['data'].tail(len(indices)), dataFrame['indice'].tail(len(indices))):
    if dataIndex in pascoaIndexes:
        ticks.append(index)
        ticksLabels.append(f"{date.day}/{date.month}/{date.year}")

for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticksLabels)

plt.show()
#endregion