pip install yfinance >> None

pip install pandas_ta >> None

pip install alphaVantage-api >> None
pip install pandas-datareader >> None

import yfinance as yf
import numpy as np
import pandas as pd
import pandas_ta as ta
import watchlist
from matplotlib.pyplot import plot as plt

def check_number_around(k, level):
  for i in range(int(k * 0.99), int(k * 1.01) + 1):
    if i % level == 0:
      return 1
  return 0

def check_integer_level(tpl, level):
  for num in tpl:
    if check_number_around(num, level) == 1:
      return 1
  return 0

def shift_values(elem, n, tpl):
  last = []
  for i in range(4):
    last += [tpl[i]] + elem[i * n : (i + 1) * n - 1 ]
  return last


def feature_generation(data, ticker, levels, n_days, index):
  #'1993-01-29','2023-11-01'

  adder = []
  m = 0
  
  last_prices = [0 for i in range(4 * n_days)]
  
  

  for ind, tpl in enumerate(data.itertuples()):
   
    if (data.loc[f'{tpl[0]}']['High'] > m):
      m = data.loc[f'{tpl[0]}']['High']
    adder.append([check_integer_level(tpl[1:5], i) for i in levels] + [tpl[4] / m - 1 ] + [i for i in last_prices])

    if (ind < n_days):
      last_prices[n_days - ind - 1] = tpl[1]
      last_prices[2 * n_days - ind - 1] = tpl[2]
      last_prices[3 * n_days - ind - 1] = tpl[3]
      last_prices[4 * n_days - ind - 1] = tpl[4]

    else:
      last_prices = shift_values(last_prices, n_days, tpl[1:5])

  			
  feature_name_last_prices = [f'{ticker} Last Open {i}' for i in range(1, n_days + 1)] + [f'{ticker} Last High {i}' for i in range(1, n_days + 1)] + [f'{ticker} Last Low {i}' for i in range(1, n_days + 1)] + [f'{ticker} Last Close {i}' for i in range(1, n_days + 1)]
  int_data = pd.DataFrame(data=np.array(adder), columns=[f'{ticker} level {i}' for i in levels] + [f'{ticker} % from high'] + feature_name_last_prices, index=index)
  return int_data


  tickers = ['^VIX', '^IXIC', 'DX-Y.NYB', '^GSPC']

tick_levels_dict = {'^GSPC': [25, 50, 100, 200, 250, 500, 1000], '^VIX': [5, 10, 20, 25, 30, 50], 'DX-Y.NYB': [5, 10, 20, 25, 30, 50, 100], '^IXIC': [50, 100, 500, 1000, 2500, 5000, 10000]}



for ticker in tickers:
  tf = "D"
  
  prices = yf.download(ticker)
  prices.to_csv(f"{ticker}_{tf}.csv")
  
  watch = watchlist.Watchlist([ticker], tf=tf, ds_name="yahoo", timed=False)
  indicators = ['squeeze', 'aberration', 'accbands', 'adosc', 'adx', 'alma', 'aroon', 'atr', 'bias', 'bop', 'cci', 'cfo', 'cg', 'cmf', 'cmo', 'cti', 
  'decreasing', 'dpo', 'ebsw', 'efi', 'entropy', 
  'eom', 'er', 'eri', 'fisher', 'hilo', 'increasing', 'inertia', 'kst', 'macd', 'massi', 'mfi', 'mom', 'natr', 'pgo', 'psar', 'psl', 'roc', 'thermo', 'ttm_trend', 'ui', 'vhf', 'vp' ]
  custom_b = ta.Strategy(name="B", ta=[{"kind": "ema", "length": 5},{"kind": "ema", "length": 10}, {"kind": "ema", "length": 20},{"kind": "ema", "length": 50},
                                        {"kind": "ema", "length": 100}, {"kind": "ema", "length": 200}, {"kind": "sma", "length": 200}, 
                                          ] + [{"kind": indicator} for indicator in indicators])
    
  watch.strategy = custom_b   #ta.AllStrategy # If you have a Custom Strategy, you can use it here.
  data = watch.load(ticker, verbose=True, )
  
  
  #data = watch.data[ticker]
  data.columns = [f'{ticker}_' + name for name in data.columns.values.tolist()]
  data = data.drop([f'{ticker}_' + 'low_Close', f'{ticker}_' + 'mean_Close', f'{ticker}_' + 'high_Close', f'{ticker}_' + 'pos_Volume', f'{ticker}_' + 'neg_Volume', f'{ticker}_' + 'total_Volume'], axis=1)
  my_features = feature_generation(prices, ticker, tick_levels_dict[ticker], 10, data.index)
  data = pd.concat([data, my_features], axis=1)

 
  data = data.loc['1993-01-29':]
 
  data.to_csv(f'{ticker}_ta_my_features.csv')


tickers = ['^VIX', '^IXIC', 'DX-Y.NYB']
all_data = pd.read_csv('^GSPC_ta_my_features.csv')
for ticker in tickers:
    data = pd.read_csv(f'{ticker}_ta_my_features.csv')
    all_data = pd.concat([all_data, data], axis=1, join='inner')

print(all_data.shape)
all_data.to_csv('data_raw.csv', index=False)


