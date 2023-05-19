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
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
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
hist = model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=2, validation_split=0.2)
end_time = time.time() - start_time
print('걸린시간 : ', end_time) # 2.4945740699768066

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss) # 0.012628240510821342
print('accuracy : ', accuracy) # 1.0 = 과적합일 가능성이 많다

# 시각화
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import font_manager, rc
font_path = 'C:/Windows\\Fonts/D2coding-Ver1.3.2-20180524-all.ttc'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='orange', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('Loss & Val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()