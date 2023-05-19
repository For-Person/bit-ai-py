# [실습] svm모델과 나의 tf keras 모델 성능 비교하기
# 1. iris
# 2. cancer
# 3. wine
# 4. california svr
import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (20640, 8) (20640,)
print('y의 라벨값 : ', np.unique(y)) 
# y의 라벨값 : [0.14999 0.175   0.225   ... 4.991   5.      5.00001]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

# Scaler 적용
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = SVR()
# model = LinearSVR()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 r2 : ', result) 

# MinMaxScaler 적용 => 결과 r2 :  0.6745809218230632
# StandardScaler 적용 => 결과 r2 :  0.7501108272937165
# MaxAbsScaler 적용 => 결과 r2 :  0.5901604818199736
# RobustScaler 적용 => 결과 r2 :  0.6873119065345796