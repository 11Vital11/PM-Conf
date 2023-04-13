import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error


import lightgbm as lgb
import joblib

import datetime as dt


from sklearn.metrics import mean_squared_error
import random
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM , Bidirectional
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

def seed(id):
    torch.manual_seed(id)
    np.random.seed(id)
    random.seed(id)
    tf.random.set_seed(id)

df = pd.read_csv('filtered_features.csv', index_col='Date')
df['Date'] = df.index
df.index = np.array(range(df.shape[0]))
df = df.drop('Unnamed: 0', axis = 1)
df.head()  

df['Date'].shape
def save_data(data, time_index_loc, name):
    data = pd.DataFrame(data)            
    data['Date'] = time_index_loc
    data.to_csv(name)

def series_data_split(data , time_index,  lag, horizon, t_train, t_val, t_test, name, step = 500):
    # data - в виде серии
     
    data = np.array(data.to_list()).reshape(-1, 1)
    indexes = np.array(range(len(data)))
    
    k = 0

    while t_val < indexes[-1]:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        x_val = []
        y_val = []

        if k == 0:
           

            scaled = np.array(data[: t_val]).reshape(1, -1)[0]
            

            
            
            
            for i in range(lag + 1, t_train + 1):
                x_train.append(np.log(scaled[i-lag:i] / scaled[i-lag-1:i-1])) 
                y_train.append(np.log(scaled[i]/scaled[i-1]))
            
            for i in range(t_train + 1, t_val):
                x_val.append(np.log(scaled[i-lag:i] / scaled[i-lag-1:i-1]))
                if i + horizon <= t_val:
                    y_val.append(np.log(scaled[i : i + horizon]/scaled[i-1:i+horizon-1]))
                else:
                    y_val.append(np.log(scaled[i : t_val]/scaled[i-1:t_val-1]))
            
            
            save_data(x_train, time_index.iloc[indexes[lag : t_train]].to_list(), f'x_train_v2_{name}_{k}.csv')
            save_data(y_train, time_index.iloc[indexes[lag : t_train]].to_list(), f'y_train_v2_{name}_{k}.csv')
            
            
            save_data(x_val, time_index.iloc[indexes[t_train: t_val - 1]].to_list(), f'x_val_v2_{name}_{k}.csv')
            save_data(y_val, time_index.iloc[indexes[t_train: t_val - 1]].to_list(), f'y_val_v2_{name}_{k}.csv')
            


        
       
        scaled_train_data = np.array(data[:t_val]).reshape(1, -1)[0]
      

        

        
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        x_val = []
        y_val = []

        
        for i in range(lag + 1, t_val):
            x_train.append(np.log(scaled_train_data[i-lag:i]/scaled_train_data[i-lag-1:i-1]))
            y_train.append(np.log(scaled_train_data[i]/scaled_train_data[i-1]))

        save_data(x_train, time_index.iloc[indexes[lag : t_val - 1]].to_list(), f'x_train_v2_{name}_{k + 1}.csv')
        save_data(y_train, time_index.iloc[indexes[lag : t_val - 1]].to_list(), f'y_train_v2_{name}_{k + 1}.csv')
        
        


        scaled_test = np.array(data[:t_test])
        scaled_test = scaled_test.reshape(1, -1)[0]

        pointer = 0
        for i in range(t_val, t_test):
            
            x_test.append(np.log(scaled_test[i-lag:i]/scaled_test[i-lag-1:i-1])) 
            if i + horizon <= t_test:
                y_test.append(np.log(scaled_test[i : i + horizon]/scaled_test[i-1:i+horizon-1]))
            else:
                y_test.append(np.log(scaled_test[i : t_test]/scaled_test[i-1 : t_test-1]))
            
              
       
        save_data(x_test, time_index.iloc[indexes[t_val  - 1: t_test - 1]].to_list(), f'x_test_v2_{name}_{k + 1}.csv')
        save_data(y_test, time_index.iloc[indexes[t_val  - 1: t_test - 1]].to_list(), f'y_test_v2_{name}_{k + 1}.csv')
          
        
        
     
        k += 1
        t_train = t_val
        

        t_val = t_test 
        t_test += step
        if t_test > indexes[-1]:
            t_test = indexes[-1] + 1

           
        
    
name = 'IXIC_Close' #'DX-Y.NYB_Close' 'VIX_Close       
ser = df[name]

time_index = df['Date']
lag = 40
horizon = 10
t_train = 1500
t_val = 1550
t_test = 1600
step = 50

series_data_split(ser , time_index, lag, horizon, t_train, t_val, t_test, name, step)



class LSTM_predictor:
    def __init__(self, input_shape, seq_num, inner_output, output_shape, num_layers, drop, bilstm = False):
        
        self.model = Sequential()
        ret_seq = True if num_layers > 1 else False
        if not bilstm:
            self.model.add(LSTM(units = inner_output, return_sequences = ret_seq, input_shape = (input_shape, seq_num)))
            self.model.add(Dropout(drop))
            if num_layers > 1:
                if num_layers > 2:
                    for i in range(num_layers-2):
                        self.model.add(LSTM(units = inner_output, return_sequences = True))
                        self.model.add(Dropout(drop))

                self.model.add(LSTM(units = inner_output))
                self.model.add(Dropout(drop))
            self.model.add(Dense(units = output_shape))
        else:
        
            # First layer of BiLSTM
            self.model.add(Bidirectional(LSTM(units = inner_output, return_sequences = ret_seq, input_shape = (input_shape, seq_num))))
            self.model.add(Dropout(drop))
            if num_layers > 1:
                if num_layers > 2:
                    for i in range(num_layers-2):
                        self.model.add(Bidirectional(LSTM(units = inner_output, return_sequences = True)))
                        self.model.add(Dropout(drop))

            
                # Second layer of BiLSTM
                self.model.add(Bidirectional(LSTM(units = inner_output)))
                self.model.add(Dropout(drop))
            self.model.add(Dense(units = output_shape))


class nn_wrapper_tf:
    def __init__(self, nn_type):
        self.type = nn_type
        
        
    def set_params(self, **params):
    
        self.num_epochs = params['num_epochs']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.hidden_size = params['hidden_size'] #, num_layers, seq_length, drop
        self.num_layers = params['num_layers']
        self.drop = params['drop']
        if 'bilstm' in params.keys():

            self.bilstm = params['bilstm']
        else:
            self.bilstm = False
        

    def fit(self, X, y, X_val = 0, y_val = 0):
        
        
        if isinstance(X, np.ndarray):
            self.X_train = np.reshape(X, (X.shape[1], X.shape[2], X.shape[0]))
            self.y_train = np.array(y).reshape((len(y), 1))
        else:
            self.X_train = np.reshape(X.values, (X.shape[1], X.shape[2], X.shape[0]))
            self.y_train = np.array(y.values).reshape((len(y), 1))
        print(X.shape[2],'lll', X.shape[0])
        # (self, input_shape, inner_output, output_shape, num_layers, drop)
        self.nn = LSTM_predictor(X.shape[2], X.shape[0], self.hidden_size, 1, self.num_layers, self.drop, self.bilstm)
        self.nn.model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        self.nn.model.fit(self.X_train, self.y_train, epochs = self.num_epochs, batch_size = self.batch_size, shuffle = False)
                
                   
    def predict(self, x):
        
        return self.nn.model.predict(np.reshape(x, (x.shape[1], x.shape[2], x.shape[0])))


import optuna


def optuna_hyper_opt(model, X_train, y_train, X_val, y_val, params, metric_weights, num_lags):
    
    
    def objective(trial):
        parameters = {}
        for i in params:
            
            if type(params[i]) == list:
                if type(params[i][0]) == int:
                    parameters.update({i: trial.suggest_int(i, params[i][0], params[i][1])})
                elif type(params[i][0]) == float:
                    parameters.update({i: trial.suggest_float(i, params[i][0], params[i][1])})
                elif type(params[i][0]) == str:
                    parameters.update({i: trial.suggest_categorical(i, params[i])})
            else:
                parameters.update({i:params[i]})
        
        print(parameters)
        
        model.set_params(**parameters)

        if isinstance(model, nn_wrapper_tf):
            model.fit(X_train, y_train, X_val, y_val)
        else:
            print('k o')
            model.fit(X_train, y_train)

        
        
        pred = horizon_forecasts_v2(model, X_val, y_val.shape[1], num_lags)
        score = np.array(scores_fun(pred, y_val))

        return np.dot(score, metric_weights)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=3, timeout=300)
    return study.best_params

def horizon_forecasts_v2(model, x, horizon, num_lags):
    
    all_features = x
    if isinstance(model, nn_wrapper_tf):
        for i in range(horizon):
            f = model.predict(all_features).reshape(-1, 1)
            
            if i == 0:
                prediction = f
            else: 
                prediction = np.concatenate((prediction, f), axis=1)
            lags = all_features[0]
            
            not_lags = all_features[1:]
            
            lags = np.concatenate((lags[:, 1:], f), axis=1)
           
            
            
            all_features = np.concatenate([np.array([lags]), not_lags])
    elif num_lags < x.shape[1]:
        not_lags = all_features[:, num_lags:]
        lags = all_features[:, :num_lags]
        for i in range(horizon):
            f = model.predict(all_features).reshape(-1, 1)
            
            if i == 0:
                prediction = f
            else: 
                prediction = np.concatenate((prediction, f), axis=1)
            
            lags = np.concatenate((lags[:, 1:], f), axis=1)
            all_features = np.concatenate((lags, not_lags), axis=1)
    else:
        lags = all_features
        for i in range(horizon):
            f = model.predict(lags).reshape(-1, 1)
            
            if i == 0:
                prediction = f
            else: 
                prediction = np.concatenate((prediction, f), axis=1)

            lags =  np.concatenate((lags[:, 1:], f), axis=1)

    return np.array(prediction)

def horizon_forecasts(model, x, horizon, num_lags):
    prediction = []
    all_features = x.reshape(1, -1)
    if num_lags < len(x):
        not_lags = all_features[0][num_lags:]
        lags = all_features[0][:num_lags]
        for i in range(horizon):
            f = model.predict(all_features)[0]
            prediction.append(f)
            lags = np.append(lags[1:], f)
            all_features = np.append(lags, not_lags).reshape(1, -1)
            
    else:
        lags = all_features
        for i in range(horizon):
            f = model.predict(lags)[0]
            prediction.append(f)
            lags = np.append(lags[0][1:], f).reshape(1, -1)
       
       

    return np.array(prediction)

def scores_fun(pred, y):
    horizon = y.shape[1]
    n = y.shape[0]
    metric_mse = 0
    metric_trend_detect = 0
    metric_weights = 0
    metric_mape = 0
    metric_true_pred = 0
    
    weights = np.array([i / horizon for i in range(1, horizon + 1)])

    for i in range(n - horizon + 1):
        diff = (pred[i] - y[i]) ** 2
        if pred[i][0] * y[i][0] > 0:
             metric_true_pred += 1
        metric_mape += mean_absolute_percentage_error(y[i], pred[i])

        metric_mse += np.sum(diff) / horizon
        
        if np.dot(np.sum(pred[i]), np.sum(y[i])) > 0:
            metric_trend_detect += 1
        
        metric_weights += np.sum(np.dot(weights, diff)) / horizon

    for ind, j in enumerate(range(n - horizon + 1, n)):
        s1 = 0
        s1_w = 0
        s_mape = 0
        s_tr_det = 0
        s_tr_det_pred = 0
        for i in range(horizon - ind - 1):
            diff = (pred[j][i] - y[j][i]) ** 2 
            s1 += diff
            if i == 0:
                if pred[j][i] * y[j][i] > 0:
                    metric_true_pred += 1
            if y[j][i] != 0:
                s_mape += np.abs(pred[j][i] - y[j][i]) / y[j][i]
            s1_w = diff * weights[i]
            s_tr_det += y[j][i]
            s_tr_det_pred += pred[j][i]
            if i == horizon - ind - 1 - 1:
                if np.dot(s_tr_det_pred, s_tr_det ) > 0:
                    metric_trend_detect += 1 - ind / horizon
        
        metric_mse += s1 / (horizon - ind - 1)
        metric_weights += s1_w / (horizon - ind -1)
        metric_mape += s_mape / (horizon - ind - 1)


    return [metric_mse/n, metric_weights/n, metric_trend_detect/n, 100 * metric_mape / n, metric_true_pred / n]


def horizon_prediction(model, X, y, k, num_lags, addit_data=0):
    time_index = X['Date']
    if isinstance(model, nn_wrapper_tf):
        X = X.drop('Date', axis = 1).values
        X = np.concatenate([np.array([X]), np.array(addit_data)])
        y = y.drop('Date', axis = 1).values
    else:
        X = X.drop('Date', axis = 1).values
        y = y.drop('Date', axis = 1).values


    hor = y.shape[1]
    pred = []
  
    
    pred = horizon_forecasts_v2(model, X, hor, num_lags)
    # pred в df превратить
    score = scores_fun(pred, y)
    
    score.append(time_index[0])
    score.append(time_index[time_index.shape[0] - 1])
  
    pred = pd.DataFrame(pred, index=time_index)
    return pred, score


def model_forecasts(model, params_set, k, HPO=False, metric_weights = np.array([0, 0, 0, 0, 0])):
    forecasts = []
    metrics = []
    df = pd.read_csv('filtered_features.csv', index_col='Date')
    df['Date'] = df.index
    df.index = np.array(range(df.shape[0]))
    df = df.drop(['Unnamed: 0'], axis = 1)
    tick = ['IXIC_Close', 'DX-Y.NYB_Close', 'VIX_Close']
    for i in range(k):

        print(i)
        if i == 0:
            data_x_train_val = pd.read_csv(f'./x_train_v2_/x_train_v2_{i}.csv').drop(['Unnamed: 0', 'Date'], axis = 1)
            num_lags = data_x_train_val.shape[1] - 1
            if isinstance(model, nn_wrapper_tf):
                
                addit_data = []
                for t in tick:
                    add_data = pd.read_csv(f'./x_train_v2_{t}_/x_train_v2_{t}_{i}.csv', index_col='Date')
                    add_data['Date'] = add_data.index
                    add_data.index = np.array(range(add_data.shape[0]))
                    add_data = add_data.drop(['Unnamed: 0', 'Date'], axis = 1)
                    
                    addit_data.append(add_data.values)
                
                data_x_train_val = np.concatenate([np.array([data_x_train_val.values]), np.array(addit_data)])
            else:
                data_x_train_val = pd.merge(data_x_train_val, df, 'inner', 'Date').drop('Date', axis=1)

            data_y_train_val = pd.read_csv(f'./y_train_v2_/y_train_v2_{i}.csv').drop(['Unnamed: 0', 'Date'], axis = 1)
            data_x_val = pd.read_csv(f'./x_val_v2_{i}.csv').drop(['Unnamed: 0'], axis = 1)
            if isinstance(model, nn_wrapper_tf):
                
                addit_data = []
                for t in tick:
                    add_data = pd.read_csv(f'./x_val_v2_{t}_{i}.csv', index_col='Date')
                    add_data['Date'] = add_data.index
                    add_data.index = np.array(range(add_data.shape[0]))
                    add_data = add_data.drop(['Unnamed: 0', 'Date'], axis = 1)
                    addit_data.append(add_data.values)
                data_x_val = np.array([data_x_val.values, addit_data])
                
            else:
                data_x_val = pd.merge(data_x_val, df, 'inner', 'Date').drop('Date', axis=1)

            data_y_val = pd.read_csv(f'y_val_v2_{i}.csv').drop(['Unnamed: 0', 'Date'], axis = 1)
        else:
            data_x_train_val = data_x_train_test
            data_y_train_val = data_y_train_test
            data_x_val = data_x_test.drop( 'Date', axis = 1)
            data_y_val = data_y_test.drop('Date', axis = 1)
        
        if HPO:
            params = optuna_hyper_opt(model, data_x_train_val.values, data_y_train_val.values, data_x_val.values, data_y_val.values, params_set, metric_weights, num_lags)
            for j in params_set:
                if j not in params.keys():
                    params.update({j : params_set[j]})    
        else:
            params = params_set 

        print(params)
        model.set_params(**params)
        

        data_x_train_test = pd.read_csv(f'./x_train_v2_/x_train_v2_{i + 1}.csv').drop(['Unnamed: 0'], axis = 1)
        if isinstance(model, nn_wrapper_tf):
            addit_data = []
            for t in tick:
                add_data = pd.read_csv(f'./x_train_v2_{t}_/x_train_v2_{t}_{i + 1}.csv', index_col='Date')
                add_data['Date'] = add_data.index
                add_data.index = np.array(range(add_data.shape[0]))
                add_data = add_data.drop(['Unnamed: 0', 'Date'], axis = 1)
                addit_data.append(add_data.values)
            
            data_x_train_test = np.concatenate([np.array([data_x_train_test.drop(['Date'], axis = 1).values]), np.array(addit_data)])
        else:
            data_x_train_test = pd.merge(data_x_train_test, df, 'inner', 'Date').drop('Date', axis=1)
            
        data_y_train_test = pd.read_csv(f'./y_train_v2_/y_train_v2_{i + 1}.csv').drop(['Unnamed: 0', 'Date'], axis = 1)
        data_x_test = pd.read_csv(f'./x_test_v2_/x_test_v2_{i + 1}.csv').drop('Unnamed: 0', axis = 1)
        if not isinstance(model, nn_wrapper_tf):
            data_x_test = pd.merge(data_x_test, df, 'inner', 'Date')
        else:
            addit_data = []
            for t in tick:
                add_data = pd.read_csv(f'./x_test_v2_{t}_/x_test_v2_{t}_{i + 1}.csv', index_col='Date')
                add_data['Date'] = add_data.index
                add_data.index = np.array(range(add_data.shape[0]))
                add_data = add_data.drop(['Unnamed: 0', 'Date'], axis = 1)
                addit_data.append(add_data.values)

        data_y_test = pd.read_csv(f'./y_test_v2_/y_test_v2_{i + 1}.csv').drop('Unnamed: 0', axis = 1)
        
        model.fit(data_x_train_test, data_y_train_test)
        res = horizon_prediction(model, data_x_test, data_y_test, i + 1, num_lags, addit_data)

        
        forecasts.append(res[0])
        metrics.append(res[1])
        print(i)
    return pd.concat(forecasts), pd.DataFrame(metrics)


    params = {
    'boosting_type': 'gbdt',
    'objective':'regression',
    'num_leaves': 100,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'metric': 'l1',
    'num_iterations': 200
}




#tf.keras.utils.disable_interactive_logging()
tf.keras.utils.enable_interactive_logging()
seed(0)

#model = lgb.LGBMRegressor()
num_layers = 1
k = 120
model_types = ['BiLSTM'] # LSTM
for j in model_types:
    bilstm = True if j == 'BiLSTM' else False
    
    for i in range(1, num_layers + 1):

       


        print(j, i)
        model = nn_wrapper_tf(j)
        #model = lgb.LGBMRegressor()
        params2 = {
        'num_epochs': 10,
        'batch_size' : 64,
        'hidden_size': 100,
        'num_layers': i,
        'drop': 0.3,
        'learning_rate': 0.05,
        'bilstm': bilstm
        }
        results = model_forecasts(model, params2, k)#, True, np.array([1, 1, 1, 1, 1]))

        results[0].to_csv(f'new_results_v6_{j}_{i}_{k}.csv')
        results[1].to_csv(f'new_results_metrics_v6_{j}_{i}_{k}.csv')


