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
model.add(Dense(14, input_dim=1))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer ='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

##### scatter 시각화
import matplotlib.pyplot as plt

plt.scatter(x, y) # 산점도 그리기
plt.plot(x, y_predict, color='blue')
plt.show()