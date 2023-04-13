import numpy as np
import pandas as pd

tick = ['^GSPC', '^IXIC', '^VIX', 'DX-Y.NYB']
model_types = ['LSTM']
for model_type in model_types:
    for num in [1]:
        df = pd.read_csv('filtered_features.csv').drop(['Unnamed: 0'], axis=1)
        model_forecasts = pd.read_csv(f'new_results_v6_{model_type}_{num}_120.csv')
        model_forecasts.head()
        model_forecasts['trend_forecast'] = model_forecasts.apply(lambda x: 1 if np.sum(x[1:]) > 0 else -1 , axis=1)
        model_forecasts['trend_forecast_next_day'] = model_forecasts.apply(lambda x: 1 if x[1] > 0 else -1 , axis=1)
        print(model_forecasts['trend_forecast'].value_counts())
        print(model_forecasts['trend_forecast_next_day'].value_counts())
        
        model_forecasts = model_forecasts.drop([f'{i}' for i in range(5, 10)], axis=1)
        model_forecasts.head()
        df = pd.merge(df, model_forecasts, 'inner', 'Date')
        df.to_csv(f'filtered_features_with_forecasts_{model_type}_{num}_120.csv')
        k = 0
        for i in tick:

            columns = [f'{i}_Open',f'{i}_High',f'{i}_Low',f'{i}_Close',f'{i}_Adj Close',
            f'{i}_Volume',f'{i}_EMA_5',f'{i}_EMA_10',f'{i}_EMA_20',f'{i}_EMA_50',f'{i}_EMA_100',f'{i}_EMA_200',f'{i}_SMA_200',
            f'{i}_MACD_12_26_9' ,f'{i} level 50', f'{i} % from high', 'Date'
            ]
            if k == 0:
                d_tick = pd.read_csv(f'{i}_ta_my_features.csv').drop(['target'], axis= 1)
                d_tick=d_tick.dropna(axis=1).reset_index(drop=True)

                df2 = pd.merge(d_tick, model_forecasts, 'inner', 'Date')
                df2.to_csv(f'{i}_ta_my_features_with_forecasts_{model_type}_{num}_120.csv')
                df2 = d_tick
                d_tick[columns].to_csv(f'{i}_some_my_features.csv')
                df3 = d_tick[columns]

                df4 = pd.merge(d_tick[columns], model_forecasts, 'inner', 'Date')
                df4.to_csv(f'{i}_some_my_features_with_forecasts_{model_type}_{num}_120.csv')
            else:
                d_tick = pd.read_csv(f'{i}_ta_my_features.csv')
                df3 = pd.merge(df3, d_tick[columns], 'inner', 'Date')
                df2 = pd.merge(df2, d_tick, 'inner', 'Date')
            k += 1

        df3 = df3.dropna(axis=1).reset_index(drop=True)
        df3.to_csv('all_some_my_features.csv')
        df3 = pd.merge(df3, model_forecasts, 'inner', 'Date')
        df3.to_csv(f'all_some_my_features_with_forecasts_{model_type}_{num}_120.csv')
        
        df2 = df2.dropna(axis=1).reset_index(drop=True)
        df2.to_csv('all_ta_features.csv')
        df2 = pd.merge(df2, model_forecasts, 'inner', 'Date')
        df2.to_csv(f'all_ta_features_with_forecasts_{model_type}_{num}_120.csv')


df = pd.read_csv('^GSPC_ta_my_features.csv')
df.head()

df2=df.dropna(axis=1).reset_index(drop=True)
df2.to_csv('^GSPC_ta_my_features.csv')