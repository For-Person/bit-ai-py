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
model = XGBRegressor(random_state=123, n_estimators=1000, 
                     learning_rate = 0.1, max_depth = 6, gamma= 1)

#3. 훈련
model.fit(x_train, y_train,  early_stopping_rounds=20, 
          eval_set = [(x_train, y_train), (x_test, y_test)], 
          eval_metric='rmse') 
# eval_metric 회귀모델 : rmse, mae, rmsle ...
#             이진분류 : error, auc, logloss ...
#             다중분류 : merror, mlogloss ...        

#4. 평가, 예측
result = model.score(x_test, y_test)

score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross validation

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

r2 = r2_score(y_test, y_predict)
print('cv pred r2 : ', r2)
# cv pred r2 :   0.780447660614057 

# SelectFromModel
from sklearn.feature_selection import SelectFromModel
thresholds = model.feature_importances_
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2:%.2f%%" 
          %(thresh, select_x_train.shape[1], score*100))
    
# (16512, 1) (4128, 1)
# Thresh=0.545, n=1, R2:44.92%
# (16512, 5) (4128, 5)
# Thresh=0.071, n=5, R2:82.91%
# (16512, 6) (4128, 6)
# Thresh=0.037, n=6, R2:83.49%
# (16512, 7) (4128, 7)
# Thresh=0.024, n=7, R2:83.35%
# (16512, 8) (4128, 8)
# Thresh=0.022, n=8, R2:82.87%
# (16512, 2) (4128, 2)
# Thresh=0.148, n=2, R2:54.73%
# (16512, 4) (4128, 4)
# Thresh=0.073, n=4, R2:82.02%
# (16512, 3) (4128, 3)
# Thresh=0.081, n=3, R2:70.76%