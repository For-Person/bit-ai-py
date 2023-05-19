import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import cifar10
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  # (10000, 32, 32, 3) (10000, 1)

# 정규화(Nomarlization) => 0~1사이로 숫자 변환
x_train, x_test = x_train/255.0, x_test/255.0

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same',
                 activation = 'relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor = 'val_loss', patience=10, mode='min',
                              verbose=1, restore_best_weights=True)
start_time = time.time()
model.fit(x_train, y_train, epochs=20, batch_size=256,
          validation_split=0.2,   
          callbacks=[earlyStopping],
          verbose=1)
end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린시간 : ', end_time)

#===========결과=============#
# epochs=20, patience=10
# loss : 0.8901200890541077
# acc : 0.7333999872207642
# 걸린시간 : 601.4675815105438
