import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.model_selection import KFold

def loadData():
  # region LOADING DATA
  df = pd.read_csv('sales-data.csv')

  # Converter a coluna de data para o tipo datetime
  df['data'] = pd.to_datetime(df['data'])

  # Extrair o ano e o mês da data
  df['ano'] = df['data'].dt.year
  df['mes'] = df['data'].dt.month

  # Criar uma nova coluna que combine ano e mês como um número único
  df['ano_mes'] = df['ano'] * 100 + df['mes']
  # Obter os índices das linhas onde ano_mes é igual a 202405 (mês está incompleto)
  index_to_drop = df[df['ano_mes'] == 202405].index
  df = df.drop(index_to_drop)
  
  # Agrupar os dados por 'ano_mes' e somar as quantidades vendidas
  df_grouped = df.groupby('ano_mes')['quantidade'].sum().reset_index()

  # Ordenar os dados pela data e resetar o índice para criar um índice sequencial
  df_grouped = df_grouped.sort_values(by='ano_mes').reset_index(drop=True)
  df_grouped['indice'] = df_grouped.index + 1  # Criar um índice sequencial começando em 1

  # Separar as variáveis independentes (X) e dependentes (y)
  X = df_grouped['indice'].values.reshape(-1, 1)
  y = df_grouped['quantidade'].values.reshape(-1, 1)
  return (X, y, df_grouped)

# endregion

# region VALIDATING DATA
def validateData(data_frame):
  # Verificar se há valores NaN ou infinitos
  if data_frame.isna().sum().iloc[0].sum() > 0:  # Verificar valores NaN
    print('there are NaN values')
    sys.exit('1')
  if np.isinf(data_frame).sum().iloc[0].sum() > 0:  # Verificar valores infinitos
    print('there are values with infinite data')
    sys.exit('1')
# endregion

# region NORMALIZING DATA
def normalizeData(X, y):
  # Calcular a média e o desvio padrão para normalizar e desnormalizar
  mean_X = np.mean(X)
  std_X = np.std(X)
  mean_y = np.mean(y)
  std_y = np.std(y)

  # Normalizar os dados
  X_normalized = (X - mean_X) / std_X
  y_normalized = (y - mean_y) / std_y
  return (X_normalized, y_normalized, mean_X, mean_y, std_X, std_y)
# endregion

def doNothing(X_train, y_train, X_test, y_test):
   {}
# region MODEL TRAINING
def evaluateModel(X_normalized, y_normalized, forEachFold = doNothing):
  # Construir o modelo de regressão
  kfold = KFold(n_splits=10, shuffle=True)
  losses = []

  for train_index, test_index in kfold.split(X_normalized):
    # Dividir os dados em conjuntos de treinamento e teste
      X_train, X_test = X_normalized[train_index], X_normalized[test_index]
      y_train, y_test = y_normalized[train_index], y_normalized[test_index]

      loss = forEachFold(X_train, y_train, X_test, y_test)
      losses.append(loss)

  # Calcular a média das perdas e desvio padrao
  mean_loss = np.mean(losses)
  std_loss = np.std(losses)
  return (losses, mean_loss, std_loss)

#endregion

#region VISUALIZING DATA
def visualizeData(model, data_frame, X, y, mean_X, mean_y, std_X, std_y, transformInput = lambda input: input):
  # Definir os meses do próximo ano para previsão
  ultimo_ano_mes = data_frame['ano_mes'].max()
  ultimo_ano = int(str(ultimo_ano_mes)[:4])
  ultimo_mes = int(str(ultimo_ano_mes)[-2:])

  # Gerar meses futuros a partir do último mês existente nos dados
  proximos_meses_ano_mes = []
  for i in range(1, 13):
      proximo_mes = ultimo_mes + i
      proximo_ano = ultimo_ano
      if proximo_mes > 12:
          proximo_ano += 1
          proximo_mes -= 12
      proximos_meses_ano_mes.append(proximo_ano * 100 + proximo_mes)

  indices_para_predicao = np.arange(1, X.max() + 13).reshape(-1, 1)

  # Normalizar os próximos índices
  indices_para_predicao_normalized = (indices_para_predicao - mean_X) / std_X

  # Fazer previsões
  previsoes_normalized = model.predict(transformInput(indices_para_predicao_normalized))
  previsoes = previsoes_normalized * std_y + mean_y

  # Mapear os índices para as labels do eixo x
  # Criar um dicionário para mapear índice para ano-mês
  index_to_date = {row['indice']: f"{str(row['ano_mes'])[4:6]}-{str(row['ano_mes'])[:4]}" for _, row in data_frame.iterrows()}
  all_dates = [index_to_date.get(i, f"{str(ano_mes)[-2:]}-{str(ano_mes)[:4]}") for i, ano_mes in zip(indices_para_predicao.flatten(), list(data_frame['ano_mes']) + proximos_meses_ano_mes)]

  # Plotar os dados e as previsões
  plt.scatter(X, y, color='blue', label='Dados Reais')
  plt.plot(indices_para_predicao, previsoes, color='red', label='Previsões')
  plt.xlabel('Mês-Ano')
  plt.ylabel('Quantidade Vendida')
  plt.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
  plt.legend()
  ticks = []
  ticks_labels = []
  for index in indices_para_predicao.flatten():
      mes = int(all_dates[index - 1][0:2])
      if mes in [3, 5]:
        ticks.append(index)
        ticks_labels.append(all_dates[index - 1])
  plt.xticks(ticks=ticks, labels=ticks_labels, rotation=90)

  plt.show()

#endregion

#region VISUALIZING LOSS
def visualizeLoss(loss):
     # Plotar os dados e as previsões
  plt.plot(loss, color='red', label='Loss')
  plt.xlabel('Fold')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()