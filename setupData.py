import pandas as pd
import numpy as np
from easterDays import getEasterDays 
import datetime as datetime

#region utils
def week_of_month(date_value):
  week = date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1
  return date_value.isocalendar()[1] if week < 0 else week
#endregion


def setupData():
  df = pd.read_csv('sales-data.csv')
  df_selic = pd.read_csv('selic-copom.csv')
  
  # Converter a coluna de data para o tipo datetime
  df['data'] = pd.to_datetime(df['data'])
  df['mes'] = df['data'].dt.month
  # df = df[~((df['data'].dt.year == 2024))] # excluindo mes de maio 2024 que está incompleto
  df = df[~((df['mes'] == 5) & (df['data'].dt.year == 2024))] # excluindo mes de maio 2024 que está incompleto

  # df = df[df['mes'].isin([3,4])] # levando em consideração apenas os dados dos meses relativos a pascoa


  # Agrupar os dados por 'data' e somar as quantidades vendidas
  df_grouped = df.groupby('data')['quantidade'].sum().reset_index()

  # Ordenar os dados pela data e resetar o índice para criar um índice sequencial
  df_grouped = df_grouped.sort_values(by='data').reset_index(drop=True)
  df_grouped['indice'] = df_grouped.index + 1  # Criar um índice sequencial começando em 1

  df_grouped['is_weekend'] = 0  # Indicar se o dia faz parte dos dias de semana aberto

  for i in range(0,7):
        df_grouped[f'weekday_{i}'] = df_grouped['data'].apply(lambda x: 1 if x.weekday() == i else 0)
        df_grouped['is_weekend'] = df_grouped['data'].apply(lambda x: 1 if x.weekday() in [5, 6] else 0)

  for i in range(1, 6):
      df_grouped[f'week_of_month_{i}'] = df_grouped['data'].apply(lambda x: 1 if week_of_month(x) == i else 0)
      
 # Adicionar colunas de vendas dos dias anteriores
  for i in range(1, 6):
      df_grouped[f'previous_day_sales_{i}'] = df_grouped['quantidade'].shift(i).fillna(0)

  # adicionando feature days_until_easter
  easterDays = getEasterDays()
  df_grouped['days_until_easter'] = np.nan
  for date in df_grouped['data']:
    future_easters = easterDays[easterDays >= date]
    if not future_easters.empty:
        next_easter = future_easters.to_series().iloc[0]
        df_grouped.loc[df_grouped['data'] == date, 'days_until_easter'] = (next_easter - date).days
    else:
        df_grouped.loc[df_grouped['data'] == date, 'days_until_easter'] = np.nan

  # adicionando feature easter_influence
  df_grouped['easter_influence'] = 0.0
  # montando array de indices da pascoa
  for easter in easterDays:
      easterIndex = df_grouped.index[df_grouped['data'] == easter].tolist()
      if not easterIndex:
            # Se a data da Páscoa não for encontrada, pegar o próximo dia disponível
            easter = easter - pd.Timedelta(days=1)
            easterIndex = df_grouped.index[df_grouped['data'] == easter].tolist()
        
      if easterIndex:
          easterIndex = easterIndex[0]
          for i in range(15):
            if easterIndex - i >= 0:
                df_grouped.loc[easterIndex - i, 'easter_influence'] = 1


  # adicionando feature taxa selic no periodo
  df_selic['conteudo/DataInicioVigencia'] = pd.to_datetime(df_selic['conteudo/DataInicioVigencia']).dt.date.astype(str)
  df_selic['conteudo/DataInicioVigencia'] = pd.to_datetime(df_selic['conteudo/DataInicioVigencia'])
  df_selic['conteudo/DataFimVigencia'] = pd.to_datetime(df_selic['conteudo/DataFimVigencia']).dt.date.astype(str)
  df_selic['conteudo/DataFimVigencia'] = pd.to_datetime(df_selic['conteudo/DataFimVigencia'])

  # Inicializar a nova coluna no dataset original
  df_grouped['selic'] = None
  df_grouped['days_since_selic_update'] = None
  # Iterar sobre cada linha do dataset original e preencher a nova coluna
  for i, row in df_grouped.iterrows():
      last_selic_update_date = None
      data = row['data']
      # Filtrar o segundo dataset para encontrar a taxa aplicável
      taxa = df_selic[(df_selic['conteudo/DataInicioVigencia'] <= data) & (df_selic['conteudo/DataFimVigencia'] >= data)]
      if not taxa.empty:
          # Se encontrar a taxa, adicionar ao dataset original
          df_grouped.at[i, 'selic'] = taxa['conteudo/TaxaSelicEfetivaAnualizada'].values[0]
          last_selic_update_date = taxa['conteudo/DataInicioVigencia'].values[0]  # Atualizar a data da última atualização da SELIC
      if taxa.empty:
          # Se nao encontrar a taxa, pega o valor da ultima encontrada
          df_grouped.at[i, 'selic'] =  df_grouped.at[i - 1, 'selic']

      # Calcular os dias desde a última atualização da SELIC
      if last_selic_update_date is not None:
            df_grouped.at[i, 'days_since_selic_update'] = (data - last_selic_update_date).days
      else:
          df_grouped.at[i, 'days_since_selic_update'] = df_grouped.at[i - 1, 'days_since_selic_update']
  df_grouped.to_csv('consolidated.csv')

setupData()