import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2

# Substitua as informações abaixo pelas suas informações de conexão
HOST = 'localhost'
USER = 'postgres'
PASSWORD = 'postgres'
DATABASE = 'postgres'
PORT = 5432

# Conectar ao banco de dados PostgreSQL
conn = psycopg2.connect(
    host=HOST,
    user=USER,
    password=PASSWORD,
    dbname=DATABASE,
    port=PORT
)

# Executar a consulta SQL e carregar os dados em um DataFrame
query = """
select count(*) * iv."Qtde" as "quantidade", 
       p."Descricao" as "produto", 
       v."Dtvenda" as "data", 
       p."Lksecao" as "secao"  
from item_venda iv 
join vendas v on v."Idvenda" = iv."Lkvenda" 
join produtos p on p."Codprod" = iv."Lkcodprod"
where p."Lksecao" = 'CHO' 
  and p."Descricao" like 'COB%' 
  and v."Dtvenda" > '2011-01-01'
group by p."Descricao", v."Dtvenda", p."Lksecao", iv."Qtde"
order by v."Dtvenda" desc;
"""

df = pd.read_sql(query, conn)

# Fechar a conexão com o banco de dados
conn.close()

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

# Verificar se há valores NaN ou infinitos
print(df_grouped.isna().sum())  # Verificar valores NaN
print(np.isinf(df_grouped).sum())  # Verificar valores infinitos

# Separar as variáveis independentes (X) e dependentes (y)
X = df_grouped['indice'].values.reshape(-1, 1)
y = df_grouped['quantidade'].values.reshape(-1, 1)

# Calcular a média e o desvio padrão para normalizar e desnormalizar
mean_X = np.mean(X)
std_X = np.std(X)
mean_y = np.mean(y)
std_y = np.std(y)

# Normalizar os dados
X_normalized = (X - mean_X) / std_X
y_normalized = (y - mean_y) / std_y

# Construir o modelo de regressão linear
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=5, activation='sigmoid'),
    tf.keras.layers.Dense(units=1)
])

# Compilar o modelo com uma taxa de aprendizado menor
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Treinar o modelo
model.fit(X_normalized, y_normalized, epochs=1000)

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
