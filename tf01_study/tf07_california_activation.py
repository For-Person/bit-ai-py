#[실습] activation 함수를 사용하여 성능 향상시키기 
#activation='relu'
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#from sklearn.datasets import load_boston #윤리적 문제로 제공안됨
from sklearn.datasets import fetch_california_housing

#1. 데이터
#datasets = load_boston()
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)
# :속성 정보:
# - 블럭 그룹의 중위수 소득
# - 블럭 그룹의 주택 연령 중위수
# - 가구당 평균 객실 수
# - 평균 가구당 침실 수
# - 모집단 블럭 그룹 모집단
# - 평균 가구원수
# - Latitude 블록 그룹 위도
# - 경도 블록 그룹 경도
# : 결측 특성 값: 없음.
print(x.shape) #(20640, 8)
print(y.shape) #(20640, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape) #(14447, 8)
print(y_train.shape) #(14447,)

#2.  모델구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=8))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation= 'linear')) # 회귀모델에서 인풋과 아웃풋 활성화함수는 'linear'

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=200)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# result
# loss : 0.4573926627635956
# r2스코어 : 0.6548167329469758
# 모든 히든레이어에 activation='relu'를 사용하면 성능이 향상됨