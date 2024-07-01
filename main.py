import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from easterDays import getEasterDays 


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

#region Loading data
def loadData():
  dataFrame = pd.read_csv('consolidated.csv')
  dataFrame['data'] = pd.to_datetime(dataFrame['data'])

  # adicionando feature days_until_easter
  easterDays = getEasterDays()
  easterIndexes = []
  for easter in easterDays:
    easterIndex = dataFrame.index[dataFrame['data'] == easter].tolist()
    if not easterIndex:
          # Se a data da Páscoa não for encontrada, pegar o próximo dia disponível
          easter = easter - pd.Timedelta(days=1)
          easterIndex = dataFrame.index[dataFrame['data'] == easter].tolist()
      
    if easterIndex:
        easterIndex = easterIndex[0]
    easterIndexes.append(easterIndex)


  # Separar as variáveis independentes (X) e dependentes (y)
  # X = dataFrame[['easter_influence', 'selic', 'days_until_easter', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'week_of_month_1', 'week_of_month_2', 'week_of_month_3', 'week_of_month_4', 'week_of_month_5']].values
  X = dataFrame[['selic', 'days_since_selic_update', 'days_until_easter', 'is_weekend', 'previous_day_sales_1', 'previous_day_sales_2', 'previous_day_sales_3', 'previous_day_sales_4', 'previous_day_sales_5']].values
  y = dataFrame['quantidade'].values.reshape(-1,  1)
  return (X, y, dataFrame, easterIndexes)

#endregion

X, y, dataFrame, easterIndexes = loadData()

#region Cross validation
innerKfold = model_selection.KFold(n_splits=10, shuffle=True)
outerKFolds = model_selection.KFold(n_splits=3, shuffle=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.16, shuffle=False)

#region parameters tuning
knn_pipe = Pipeline(steps=[
  ('scaler', MinMaxScaler()),
  ('knn', KNeighborsRegressor()),
])
knnGridSearch = model_selection.GridSearchCV(
    estimator=knn_pipe,
    param_grid={'knn__n_neighbors': list(np.arange(1, 15)), 'knn__weights':['uniform', 'distance'], 'knn__metric': ['manhattan', 'euclidean', 'minkowski']},
    scoring='neg_mean_absolute_error',
    cv=outerKFolds,
    refit=True,
    error_score='raise'
)
knnGridSearch.fit(X_train, y_train)
knnBestParams = knnGridSearch.best_params_

mlp_pipe = Pipeline(steps=[
  ('scaler', MinMaxScaler()),
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

linear_pipe = Pipeline(steps=[
  ('scaler', MinMaxScaler()),
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

poly_pipe = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('poly', PolynomialFeatures()),
    ('model', LinearRegression()),
])
polyGridSearch = model_selection.GridSearchCV(
    estimator=poly_pipe,
    param_grid={'poly__degree': np.arange(1, 5), 'poly__include_bias': [False]},
    scoring='neg_mean_absolute_error',
    cv=outerKFolds,
    refit=True
)
polyGridSearch.fit(X_train, y_train)
polyBestParams = polyGridSearch.best_params_

svr_pipe = Pipeline(steps=[
  ('scaler', MinMaxScaler()),
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

print(f"KNN best params: {knnBestParams}")
print(f"MLP best params: {mlpBestParams}")
print(f"Linear best params: {linearBestParams}")
print(f"Poly best params: {polyBestParams}")
print(f"svr best params: {svrBestParams}")

knnErrorsMAE = []
knnErrorsMSE = []
knnErrorsR2 = []
mlpErrorsMAE = []
mlpErrorsMSE = []
mlpErrorsR2 = []
linearRegressionErrorsMAE = []
linearRegressionErrorsMSE = []
linearRegressionErrorsR2 = []
polyErrorsMAE = []
polyErrorsMSE = []
polyErrorsR2 = []
svrErrorsMAE = []
svrErrorsMSE = []
svrErrorsR2 = []

for trainIndexes, testIndexes in innerKfold.split(X, y):
  normalizer = MinMaxScaler()
  X_train_internal, y_train_internal = X[trainIndexes], y[trainIndexes]
  normalizer.fit(X_train_internal)
  X_train_internal = normalizer.transform(X_train_internal)
  X_test_internal, y_test_internal = X[testIndexes], y[testIndexes]
  X_test_internal = normalizer.transform(X_test_internal)

  knn = KNeighborsRegressor(**remove_prefix(knnBestParams))
  knn.fit(X_train_internal, y_train_internal)
  knn_predictions = knn.predict(X_test_internal)
  knnErrorMAE = metrics.mean_absolute_error(y_test_internal, knn_predictions)
  knnErrorMSE = metrics.mean_squared_error(y_test_internal, knn_predictions)
  knnErrorR2 = metrics.r2_score(y_test_internal, knn_predictions)
  knnErrorsMAE.append(knnErrorMAE)
  knnErrorsMSE.append(knnErrorMSE)
  knnErrorsR2.append(knnErrorR2)

  mlp = MLPRegressor(**remove_prefix(mlpBestParams))
  mlp.fit(X_train_internal, y_train_internal.ravel())
  mlp_predictions = mlp.predict(X_test_internal)
  mlpErrorMAE = metrics.mean_absolute_error(y_test_internal, mlp_predictions)
  mlpErrorMSE = metrics.mean_squared_error(y_test_internal, mlp_predictions)
  mlpErrorR2 = metrics.r2_score(y_test_internal, mlp_predictions)
  mlpErrorsMAE.append(mlpErrorMAE)
  mlpErrorsMSE.append(mlpErrorMSE)
  mlpErrorsR2.append(mlpErrorR2)

  linearRegression = LinearRegression(**remove_prefix(linearBestParams))
  linearRegression.fit(X_train_internal, y_train_internal)
  linear_predictions = linearRegression.predict(X_test_internal)
  linearErrorMAE = metrics.mean_absolute_error(y_test_internal, linear_predictions)
  linearErrorMSE = metrics.mean_squared_error(y_test_internal, linear_predictions)
  linearRegressionErrorR2 = metrics.r2_score(y_test_internal, linear_predictions)
  linearRegressionErrorsMAE.append(linearErrorMAE)
  linearRegressionErrorsMSE.append(linearErrorMSE)
  linearRegressionErrorsR2.append(linearRegressionErrorR2)


  polyModel = LinearRegression()
  polyFeat = PolynomialFeatures(**remove_prefix(polyBestParams))
  poly_features = polyFeat.fit_transform(X_train_internal)
  polyModel.fit(poly_features, y_train_internal)
  poly_predictions = polyModel.predict(polyFeat.fit_transform(X_test_internal))
  polyErrorMAE = metrics.mean_absolute_error(y_test_internal, poly_predictions)
  polyErrorMSE = metrics.mean_squared_error(y_test_internal, poly_predictions)
  polyErrorR2 = metrics.r2_score(y_test_internal, poly_predictions)
  polyErrorsMAE.append(polyErrorMAE)
  polyErrorsMSE.append(polyErrorMSE)
  polyErrorsR2.append(polyErrorR2)

  svr = SVR(**remove_prefix(svrBestParams))
  svr.fit(X_train_internal, y_train_internal.ravel())
  svr_predictions = svr.predict(X_test_internal)
  svrErrorMAE = metrics.mean_absolute_error(y_test_internal, svr_predictions)
  svrErrorMSE = metrics.mean_squared_error(y_test_internal, svr_predictions)
  svrErrorR2 = metrics.r2_score(y_test_internal, svr_predictions)
  svrErrorsMAE.append(svrErrorMAE)
  svrErrorsMSE.append(svrErrorMSE)
  svrErrorsR2.append(svrErrorR2)

validationMetrics = {
    'Model': ['KNN', 'MLP', 'Linear', 'Poly', 'SVR'],
    'Mean MAE': [
        np.mean(knnErrorsMAE),
        np.mean(mlpErrorsMAE),
        np.mean(linearRegressionErrorsMAE),
        np.mean(polyErrorMAE),
        np.mean(svrErrorMAE),
    ],
    'Mean MSE': [
        np.mean(knnErrorsMSE),
        np.mean(mlpErrorsMSE),
        np.mean(linearRegressionErrorsMSE),
        np.mean(polyErrorMSE),
        np.mean(svrErrorMSE),
    ],
    'Mean R2': [
        np.mean(knnErrorsR2),
        np.mean(mlpErrorsR2),
        np.mean(linearRegressionErrorsR2),
        np.mean(polyErrorR2),
        np.mean(svrErrorR2),
    ]
}

# Make predictions
knn_predictions = knnGridSearch.predict(X_test)
linear_predictions = linearGridSearch.predict(X_test)
mlp_predictions = mlpGridSearch.predict(X_test)
poly_predictions = polyGridSearch.predict(X_test)
svr_predictions = svrGridSearch.predict(X_test)

# Create a DataFrame to store the predictions and metrics
predictionDataFrame = pd.DataFrame({
    'y_test': y_test.flatten(),
    'knn_predictions': knn_predictions.flatten(),
    'mlp_predictions': mlp_predictions.flatten(),
    'linear_predictions': linear_predictions.flatten(),
    'poly_predictions': poly_predictions.flatten(),
    'svr_predictions': svr_predictions.flatten()
})

# Calculate metrics for each model
predictionMetrics = {
    'Model': ['KNN', 'MLP', 'Linear', 'Poly', 'SVR'],
    'MAE': [
        metrics.mean_absolute_error(y_test, knn_predictions),
        metrics.mean_absolute_error(y_test, mlp_predictions),
        metrics.mean_absolute_error(y_test, linear_predictions),
        metrics.mean_absolute_error(y_test, poly_predictions),
        metrics.mean_absolute_error(y_test, svr_predictions)
    ],
    'MSE': [
        metrics.mean_squared_error(y_test, knn_predictions),
        metrics.mean_squared_error(y_test, mlp_predictions),
        metrics.mean_squared_error(y_test, linear_predictions),
        metrics.mean_squared_error(y_test, poly_predictions),
        metrics.mean_squared_error(y_test, svr_predictions)
    ],
    'R2': [
        metrics.r2_score(y_test, knn_predictions),
        metrics.r2_score(y_test, mlp_predictions),
        metrics.r2_score(y_test, linear_predictions),
        metrics.r2_score(y_test, poly_predictions),
        metrics.r2_score(y_test, svr_predictions)
    ]
}

# Create a DataFrame for validation_metrics
validationMetrics = pd.DataFrame(validationMetrics)

# Create a DataFrame for metrics
predictionMetrics = pd.DataFrame(predictionMetrics)

# Save predictions and metrics to CSV files
validationMetrics.to_csv('validation-metrics.csv', index=False)
predictionMetrics.to_csv('predictions-metrics.csv', index=False)
predictionDataFrame.to_csv('predictions.csv', index=False)

#endregion
#endregion

# Plotar os dados e as previsões

indices = range(len(X_test))
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.tight_layout(pad=5.0)

y_min = min(y_test.min(), predictionDataFrame.min().min())
y_max = max(y_test.max(), predictionDataFrame.max().max()) + 20

# Dados Reais
axs[0, 0].scatter(indices, y_test, color='blue', label='Real data', marker="o", alpha=0.3)
axs[0, 0].set_title('Real data')
axs[0, 0].set_ylabel('Amount sold')
axs[0, 0].set_ylim(y_min, y_max)
axs[0, 0].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[0, 0].legend()
axs[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões KNN
axs[0, 1].scatter(indices, predictionDataFrame['knn_predictions'], color='gray', label='Predicted KNN', marker="o", alpha=0.3)
axs[0, 1].set_title('Predicted KNN')
axs[0, 1].set_ylabel('Amount sold')
axs[0, 1].set_ylim(y_min, y_max)
axs[0, 1].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[0, 1].legend()
axs[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões MLP
axs[1, 0].scatter(indices, predictionDataFrame['mlp_predictions'], color='red', label='Predicted MLP', marker="o", alpha=0.3)
axs[1, 0].set_title('Predicted MLP')
axs[1, 0].set_ylabel('Amount sold')
axs[1, 0].set_ylim(y_min, y_max)
axs[1, 0].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[1, 0].legend()
axs[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões Linear
axs[1, 1].scatter(indices, predictionDataFrame['linear_predictions'], color='green', label='Predicted Linear', marker="o", alpha=0.3)
axs[1, 1].set_title('Predicted Linear')
axs[1, 1].set_ylabel('Amount sold')
axs[1, 1].set_ylim(y_min, y_max)
axs[1, 1].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[1, 1].legend()
axs[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões Poly
axs[2, 0].scatter(indices, predictionDataFrame['poly_predictions'], color='orange', label='Predicted Poly', marker="o", alpha=0.3)
axs[2, 0].set_title('Predicted Poly')
axs[2, 0].set_ylabel('Amount sold')
axs[2, 0].set_ylim(y_min, y_max)
axs[2, 0].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[2, 0].legend()
axs[2, 0].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Previsões SVR
axs[2, 1].scatter(indices, predictionDataFrame['svr_predictions'], color='purple', label='Predicted SVR', marker="o", alpha=0.3)
axs[2, 1].set_title('Predicted SVR')
axs[2, 1].set_ylabel('Amount sold')
axs[2, 1].set_ylim(y_min, y_max)
axs[2, 1].grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
axs[2, 1].legend()
axs[2, 1].axhline(y=0, color='black', linestyle='--', linewidth=1.2)  # Linha tracejada em y = 0

# Adicionando ticks e labels
ticks = []
ticksLabels = []
for index, date, dataIndex in zip(indices, dataFrame['data'].tail(len(indices)), dataFrame['indice'].tail(len(indices))):
    if dataIndex in easterIndexes:
        ticks.append(index)
        ticksLabels.append(f"{date.day}/{date.month}/{date.year}")

for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticksLabels)

plt.show()
#endregion