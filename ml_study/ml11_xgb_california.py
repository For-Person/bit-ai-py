import numpy as np
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

print(x.shape, y.shape) # (150, 4) (150,)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 :  [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, shuffle=True
)
# train_size 안에 validation도 포함되어있다.

#kfold
n_splits = 11
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Scaler 적용 
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross validation

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

r2 = r2_score(y_test, y_predict)
print('cv pred r2 : ', r2)
# cv pred r2 :   0.780447660614057 => XGB 적용 더 나빠짐