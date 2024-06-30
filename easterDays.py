import pandas as pd
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
    '2025-04-20'
              ]
easterDays = pd.to_datetime(easterDays)

def getEasterDays():
  return easterDays