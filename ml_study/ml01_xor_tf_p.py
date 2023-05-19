import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) #sklearn의 Perceptron과 동일

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가, 예측
loss, acc = model.evaluate(x_data, y_data)
print('모델 result : ', loss)

y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_predict)

print('acc : ', acc)

# 모델 result :  0.8293380737304688
# 1/1 [==============================] - 0s 64ms/step
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [[0.46258852]
#  [0.6615811 ]
#  [0.7309175 ]
#  [0.860513  ]]
# acc :  0.75