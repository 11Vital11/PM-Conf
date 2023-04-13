pip install ipynb 

import import_ipynb

import numpy as np
import pandas as pd
import featuretools as ft
from featuretools import selection


from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, mutual_info_classif
data = pd.read_csv('data_raw.csv')
data = data.drop(['Date.1', 'Date.2', 'Date.3'], axis=1)
data.shape
data.head()

filtered = selection.remove_highly_correlated_features(data, pct_corr_threshold=0.98)
filtered = selection.remove_low_information_features(filtered)
filtered = selection.remove_highly_null_features(filtered)
filtered = selection.remove_single_value_features(filtered)
tickers = ['^VIX', '^IXIC', 'DX-Y.NYB', '^GSPC']
filtered['^GSPC_High'] = data['^GSPC_High']
filtered['^GSPC_Low'] = data['^GSPC_Low']
filtered['^GSPC_Close'] = data['^GSPC_Close']
filtered['^VIX_Close'] = data['^VIX_Close']
filtered['^IXIC_Close'] = data['^IXIC_Close']
filtered['^DX-Y.NYB_Close'] = data['^DX-Y.NYB_Close']
filtered = filtered.dropna()
filtered.head()
filtered.shape, data.shape
filtered.to_csv('filtered_features.csv')