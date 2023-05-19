#1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
# from tensorflow.python,keras.models import Sequential
# from tensorflow.python,keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1)) #입력층
model.add(Dense(10))             #히든레이어
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))              #출력층

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000) # epoch:전체 데이터셋을 몇 번 반복학습할지 설정

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([4])
print('4의 예측값 : ', result)