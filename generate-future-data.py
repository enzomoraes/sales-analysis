import pandas as pd
from datetime import datetime, timedelta
from easterDays import getEasterDays

# Função para calcular a data da Páscoa
def get_next_easter_date(date):
    easter_dates = {d.year: d for d in getEasterDays()}
    if date <= easter_dates[date.year]:
      return easter_dates[date.year]
    else:
      return easter_dates[date.year + 1]

# Função para gerar os dados a partir de uma data inicial
def generate_data_from_last_row(start_date, initial_selic_rate, initial_days_since_selic_update, num_records):
    current_date = start_date
    data = []
    selic_rate = initial_selic_rate
    days_since_selic_update = initial_days_since_selic_update
    decrement_period = 40  # Aproximadamente 3 meses de dias úteis
    
    while len(data) < num_records:
        easter = get_next_easter_date(current_date)
        days_until_easter = (easter - current_date).days
        days_since_selic_update += 1

        # Adiciona somente dias da semana, e domingos quando a Páscoa está próxima
        if (current_date.weekday() < 5 or (0 <= days_until_easter < 30 and current_date.weekday() == 6)) and days_until_easter >= 0:
            is_weekend = 1 if current_date.weekday() >= 5 else 0

            data.append({
                'data': current_date.strftime('%Y-%m-%d'),
                'selic': round(selic_rate, 2),
                'days_since_selic_update': days_since_selic_update,
                'days_until_easter': days_until_easter,
                'is_weekend': is_weekend,
            })
            # Atualiza a taxa Selic a cada 40 dias úteis
            if days_since_selic_update >= decrement_period:
                selic_rate -= 0.25
                days_since_selic_update = 0

        current_date += timedelta(days=1)

    return pd.DataFrame(data)

# Lê o arquivo CSV existente
df_existing = pd.read_csv('consolidated.csv')

# Obtém a última linha do DataFrame existente
last_row = df_existing.iloc[-1]
start_date = datetime.strptime(last_row['data'], '%Y-%m-%d') + timedelta(days=1)
initial_selic_rate = last_row['selic']
initial_days_since_selic_update = last_row['days_since_selic_update']

# Gera 700 novos registros a partir da última data
df_new = generate_data_from_last_row(start_date, initial_selic_rate, initial_days_since_selic_update, 700)

# Concatena os dados existentes com os novos dados
df_combined = pd.concat([df_existing, df_new], ignore_index=True)

# Salva o DataFrame combinado em um novo CSV
df_combined.to_csv('future.csv', index=False)