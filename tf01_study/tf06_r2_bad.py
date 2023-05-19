# [실습]
#1. R2 를 음수가 아닌 0.5이하로 만들어보기
#2. 데이터는 건드리지 마세요
#3. 레이어는 input, output 포함 7개 이상(히든레이어 5개 이상)으로 만들기
#4. batch_size=1
#5. 히든레이어의 노드 갯수는 10개 이상 100개 이하 => Dense 옆에 있는 괄호안에 숫자
#6. train_size = 0.7
#7. epochs=100 이상

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, # x, y 사이즈 
    test_size=0.3, # test의 사이즈 보통 30% 
    train_size=0.7, # train의 사이즈는 보통 70%임
    random_state=100, # 데이터를 난수값에 의해 추출한다는 의미이며, 중요한 하이퍼파라미터임
    shuffle=True # 데이터를 섞어서 가지고 올 것인지를 정함
)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer ='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

##### R2score
from sklearn.metrics import r2_score, accuracy_score
r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)

# result(1)
# loss : 25.10774040222168
# r2스코어 : 0.6024091950172759
# 노드수를 늘리고 히든레이어의 갯수를 늘리면 r2 score가 좋지 않음
