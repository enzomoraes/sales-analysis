import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sys

# region LOADING DATA
df = pd.read_csv('sales-data.csv')

# Converter a coluna de data para o tipo datetime
df['data'] = pd.to_datetime(df['data'])

# Extrair o ano e o mês da data
df['ano'] = df['data'].dt.year
df['mes'] = df['data'].dt.month

# Criar uma nova coluna que combine ano e mês como um número único
df['ano_mes'] = df['ano'] * 100 + df['mes']

# Agrupar os dados por 'ano_mes' e somar as quantidades vendidas
df_grouped = df.groupby('ano_mes')['quantidade'].sum().reset_index()

# Ordenar os dados pela data e resetar o índice para criar um índice sequencial
df_grouped = df_grouped.sort_values(by='ano_mes').reset_index(drop=True)
df_grouped['indice'] = df_grouped.index + 1  # Criar um índice sequencial começando em 1

# Criar um dicionário para mapear índice para ano-mês
index_to_date = {row['indice']: f"{str(row['ano_mes'])[4:6]}-{str(row['ano_mes'])[:4]}" for _, row in df_grouped.iterrows()}

# Separar as variáveis independentes (X) e dependentes (y)
X = df_grouped['indice'].values.reshape(-1, 1)
y = df_grouped['quantidade'].values.reshape(-1, 1)

# endregion
# region VALIDATING DATA

# Verificar se há valores NaN ou infinitos
if df_grouped.isna().sum()[1].sum() > 0:  # Verificar valores NaN
  print('there are NaN values')
  sys.exit('1')
if np.isinf(df_grouped).sum()[1].sum() > 0:  # Verificar valores infinitos
  print('there are values with infinite data')
  sys.exit('1')
  

# endregion
# region NORMALIZING DATA

# Calcular a média e o desvio padrão para normalizar e desnormalizar
mean_X = np.mean(X)
std_X = np.std(X)
mean_y = np.mean(y)
std_y = np.std(y)

# Normalizar os dados
X_normalized = (X - mean_X) / std_X
y_normalized = (y - mean_y) / std_y
# endregion

# region MODEL TRAINING
# Construir o modelo de regressão
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Lambda(lambda x: tf.concat([x, x**2, x**3], axis=1)),
    tf.keras.layers.Dense(units=5, activation='sigmoid'),
    tf.keras.layers.Dense(units=1)
])

# Compilar o modelo com uma taxa de aprendizado menor
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Treinar o modelo
model.fit(X_normalized, y_normalized, epochs=500)

# Definir os meses do próximo ano para previsão
ultimo_ano_mes = df_grouped['ano_mes'].max()
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
previsoes_normalized = model.predict(indices_para_predicao_normalized)
previsoes = previsoes_normalized * std_y + mean_y

# Mapear os índices para as labels do eixo x
# all_indices = np.append(X, indices_para_predicao)
all_dates = [index_to_date.get(i, f"{str(ano_mes)[-2:]}-{str(ano_mes)[:4]}") for i, ano_mes in zip(indices_para_predicao.flatten(), list(df_grouped['ano_mes']) + proximos_meses_ano_mes)]

# Exibir as previsões
for mes, venda, index in zip(indices_para_predicao, previsoes, proximos_meses_ano_mes):
    print(f"Previsão de vendas para o mês {str(index)[-2:]}-{str(index)[:4]}: {venda[0]:.2f}")

#endregion

#region VISUALIZING DATA
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
