import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000) # 임베딩 레이어의 input_dim
print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(x_test.shape, y_test.shape) # (25000,) (25000,)
print(np.unique(y_train, return_counts=True)) # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
print(len(np.unique(y_train))) # 2 => 긍정 부정 2개

# 최대길이와 평균길이
print('리뷰의 최대 길이 : ', max(len(i) for i in x_train))
print('리뷰의 평균 길이 : ', sum(map(len, x_train)) / len(x_train))
# 리뷰의 최대 길이 :  2494
# 리뷰의 평균 길이 :  238.71364

# pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100)
x_test = pad_sequences(x_test, padding='pre', maxlen=100)

print(x_train.shape, y_train.shape) # (25000, 100) (25000, 2)
print(x_test.shape, y_test.shape) # (25000, 100) (25000, 2)

# 모델 구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100)) 
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# [실습] 코드 완성하기
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min',
                              verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, 
                      filepath='./_mcp/tf21_imdb.hdf5')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, 
          validation_split=0.2, callbacks=[earlyStopping, mcp], verbose=1)
end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린시간 : ', end_time)

# loss : 0.5700131058692932 
# acc : 0.697359979152679
# 걸린시간 : 1166.4360191822052 
