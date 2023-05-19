import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)
print(df.head())
print(df['Population'])

df['Population'] = np.log1p(df['Population'])
print(df['Population'].head())
