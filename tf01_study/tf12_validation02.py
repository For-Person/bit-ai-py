import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

#x = np.array(range(1, 21))
#y = np.array(range(1, 21))
#print(x.shape)
#print(y.shape)
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

x_test = np.array([17, 18, 19, 20])
y_test = np.array([17, 18, 19, 20])

x_val = np.array([13, 14, 15, 16])
y_val = np.array([13, 14, 15, 16])

#2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim=1))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer ='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=[x_val, y_val])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([21])
print('21의 예측값 : ', result)

## history_val_loss 출력
print('=========================================')
print(hist)
#print(hist.history)
print(hist.history['val_loss'])

## loss와 val_loss 시각화
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import font_manager, rc
font_path = 'C:/Windows\\Fonts/D2coding-Ver1.3.2-20180524-all.ttc'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('loss & val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
