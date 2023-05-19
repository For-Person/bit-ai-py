import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 :  [0 1]

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
from lightgbm import LGBMClassifier
model = LGBMClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross validation
print('cv acc : ', score )

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)
# cv pred acc :  0.9473684210526315 => xgb 결과
# cv pred acc :  0.9473684210526315 => lgbm 결과