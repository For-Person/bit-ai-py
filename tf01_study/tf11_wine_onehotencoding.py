# [실습] one hot encoding을 사용하여 분석
# (세 가지 방법중 하나 사용)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import time

#1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

### one hot encoding
# 1. keras.utils의 to_categorical
from keras.utils import to_categorical

y = to_categorical(y)
print(y)
print(y.shape) #(178, 3)

# 2. pandas의 get_dummies
# 3. sklearn.preprocessing의 onehotcoding 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape) # (124, 13) (124, 3)
print(x_test.shape, y_test.shape)   # (54, 13) (54, 3)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=2)
end_time = time.time() - start_time
print('걸린시간 : ', end_time)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss) # 0.09724389016628265
print('accuracy : ', accuracy) # 0.9629629850387573
