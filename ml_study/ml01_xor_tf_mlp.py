import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#[실습] MLP 모델 구성하여 acc 1.0 만들기
#2. 모델
model = Sequential()
model.add(Dense(32, input_dim=2)) #MLP(multi layer perceptron)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가, 예측
loss, acc = model.evaluate(x_data, y_data)
print('모델 result : ', loss)

y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_predict)

print('acc : ', acc)

# 모델 result :  0.008711216039955616
# 1/1 [==============================] - 0s 82ms/step
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [[0.0150164 ]
#  [0.9925065 ]
#  [0.99293196]
#  [0.00508679]]
# acc :  1.0