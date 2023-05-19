import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
import time

#1. 데이터
datasets = load_diabetes()
print(datasets.DESCR)
print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape) # (309, 10) (309,)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# result
# loss : 2677.436767578125
# r2스코어 : 0.4947661394130811