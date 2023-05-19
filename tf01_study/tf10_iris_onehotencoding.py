import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import time

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data # x = datasets['data'] 동일
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

### one hot encoding
from keras.utils import to_categorical

y = to_categorical(y)
print(y)
print(y.shape) #(150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
print(x_train.shape, y_train.shape) # (105, 4) (105,)
print(x_test.shape, y_test.shape)   # (45, 4) (45,)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=4))
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
print('걸린시간 : ', end_time) # 2.4945740699768066

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss) # 0.012628240510821342
print('accuracy : ', accuracy) # 1.0 = 과적합일 가능성이 많다