import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

#1. 데이터
datasets = load_digits()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)

from keras.utils import to_categorical

y = to_categorical(y)
print(y)
print(y.shape) # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

print(x_train.shape, y_train.shape) # (1257, 64) (1257, 10)
print(x_test.shape, y_test.shape)   # (540, 64) (540, 10)
# print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=64))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='softmax'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=2)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss) # 0.2277863472700119
print('accuracy : ', accuracy) # 0.9777777791023254
