import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

from easterDays import getEasterDays

# Carregar os modelos
knnModel = joblib.load('./models/knn.pkl')
linearModel = joblib.load('./models/linear.pkl')
mlpModel = joblib.load('./models/mlp.pkl')
polynomialModel = joblib.load('./models/polynomial.pkl')

# Carregar o dataset
dataFrame = pd.read_csv('future.csv')

# # Adicionar manualmente 5 linhas de dados anteriores de 2023
# manual_data = [
#     # 'data', 'days_until_easter', 'is_weekend', 'selic', 'days_since_selic_update', 'quantidade', 'previous_day_sales_1', 'previous_day_sales_2', 'previous_day_sales_3', 'previous_day_sales_4', 'previous_day_sales_5'
#     ['2023-12-22', 95, 0, 11.65, 39, 25.0, 11.0, 10.0, 18.0, 4.0, 16.0],
#     ['2023-12-26', 94, 0, 11.65, 40, 7.0, 25.0, 11.0, 10.0, 18.0, 4.0],
#     ['2023-12-27', 93, 0, 11.65, 41, 3.0, 7.0, 25.0, 11.0, 10.0, 18.0],
#     ['2023-12-28', 92, 0, 11.65, 42, 2.0, 3.0, 7.0, 25.0, 11.0, 10.0],
#     ['2023-12-29', 91, 0, 11.65, 43, 3.0, 2.0, 3.0, 7.0, 25.0, 11.0],
# ]

# manual_df = pd.DataFrame(manual_data, columns=['data', 'days_until_easter', 'is_weekend', 'selic', 'days_since_selic_update', 'quantidade', 'previous_day_sales_1', 'previous_day_sales_2', 'previous_day_sales_3', 'previous_day_sales_4', 'previous_day_sales_5'])

# dataFrame = pd.concat([manual_df, dataFrame], ignore_index=True)
dataFrame['data'] = pd.to_datetime(dataFrame['data'])

startIndex = dataFrame.isnull().any(axis=1).idxmax()

# Função para prever vendas e preencher colunas de vendas anteriores
def predict_and_fill_sales(df):
    for i in range(startIndex, len(df)):
        previous_day_sales_1 = df.loc[i-1:i-1:-1, 'previous_day_sales_1'].values.flatten()
        previous_day_sales_2 = df.loc[i-1:i-1:-1, 'previous_day_sales_2'].values.flatten()
        previous_day_sales_3 = df.loc[i-1:i-1:-1, 'previous_day_sales_3'].values.flatten()
        previous_day_sales_4 = df.loc[i-1:i-1:-1, 'previous_day_sales_4'].values.flatten()
        amount = df.loc[i-1:i-1:-1, 'quantidade'].values.flatten()

        # Adicionar as vendas anteriores nas colunas correspondentes
        df.loc[i, 'previous_day_sales_1'] = amount
        df.loc[i, 'previous_day_sales_2'] = previous_day_sales_1
        df.loc[i, 'previous_day_sales_3'] = previous_day_sales_2
        df.loc[i, 'previous_day_sales_4'] = previous_day_sales_3
        df.loc[i, 'previous_day_sales_5'] = previous_day_sales_4

        
        # Preparar os dados para previsão
        X = df.loc[i, ['selic', 'days_since_selic_update', 'days_until_easter', 'is_weekend', 
                       'previous_day_sales_1', 'previous_day_sales_2', 'previous_day_sales_3', 
                       'previous_day_sales_4', 'previous_day_sales_5']].values.reshape(1, -1)
        
        # Prever as vendas com os modelos
        knn_pred = knnModel.predict(X)[0]
        linear_pred = linearModel.predict(X)[0]
        mlp_pred = mlpModel.predict(X)[0]
        polynomial_pred = polynomialModel.predict(X)[0]
        
        # Adicionar a previsão ao dataframe
        df.loc[i, 'quantidade'] = knn_pred
        
    return df

# Preencher as previsões no dataframe
dataFrame = predict_and_fill_sales(dataFrame)
print(f'vendas de chocolate de pascoa', dataFrame[~(dataFrame['data'].dt.year != 2024)]['quantidade'].sum())
# Remover as linhas manuais adicionadas no início
dataFrame = dataFrame.iloc[5:].reset_index(drop=True)

# Salvar o DataFrame atualizado em um novo CSV
dataFrame.to_csv('future_predictions.csv', index=False)

easterDays = getEasterDays()
ticks = []
ticksLabels = []
for index, date in zip(range(len(dataFrame)), dataFrame['data']):
  if date in easterDays:
    ticks.append(index - 5)
    ticksLabels.append(f"{date.day}/{date.month}/{date.year}")

# Plotar o gráfico
plt.figure(figsize=(10, 5))
plt.scatter(range(len(dataFrame)), dataFrame['quantidade'], marker='o', linestyle='-', color='b', alpha=0.3)
plt.fill_between(range(len(dataFrame))[startIndex:len(dataFrame)], [500], [-10], color='skyblue', alpha=0.3, label='Predictions')
plt.xlabel('Data')
plt.ylabel('Quantidade')
plt.title('Previsão de Vendas')
plt.xticks(rotation=45, ticks=ticks, labels=ticksLabels)
plt.tight_layout()
plt.show()