import numpy as np
from keras.preprocessing.text import Tokenizer

#1. 데이터
docs = ['재밌었요', '재미없다', '돈 아깝다', '숙면했어요', '최고에요', '꼭 봐라',
        '세 번 봐라', '또 보고싶다', 'n회차 관람', '배우가 정말 이쁘긴 했어요',
        '발연기에요', '추천해요', '최악', '후회', '돈 버렸다', '글쎄요', 
        '보다 나왔다', '망작이다', '연기가 어색해요', '차라리 기부할걸', 
        '다움편 나왔으면 좋겠다', '다른 거 볼걸', '감동이다']
# 긍정 1, 부정 0
labels = np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 1])

# Tokenizer
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'돈': 1, '봐라': 2, '재밌었요': 3, '재미없다': 4, '아깝다': 5, '숙면했어요': 6, '최
# 고에요': 7, '꼭': 8, '세': 9, '번': 10, '또': 11, '보고싶다': 12, 'n회차': 13, '관람': 14, '배우가': 15, '정말': 16, '이쁘긴': 17, '했어요': 18, '발연기에요': 19, '추천해
# 요': 20, '최악': 21, '후회': 22, '버렸다': 23, '글쎄요': 24, '보다': 25, '나왔다': 26, '망작이다': 27, '연기가': 28, '어색해요': 29, '차라리': 30, '기부할걸': 31, '다움편
# ': 32, '나왔으면': 33, '좋겠다': 34, '다른': 35, '거': 36, '볼걸': 37, '감동이다': 38}

x = token.texts_to_sequences(docs)
print(x)
# [[3], [4], [1, 5], [6], [7], [8, 2], [9, 10, 2], [11, 12], [13, 14], 
#  [15, 16, 17, 18], [19], [20], [21], [22], [1, 23], [24], [25, 26], 
#  [27], [28, 29], [30, 31], [32, 33, 34], [35, 36, 37], [38]]

# pad_sequences
from keras_preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=4)
print(pad_x)
print(pad_x.shape) # (23, 4) => 4는 maxlen

word_size = len(token.word_index)
print('word_size : ', word_size) # word_size :  38

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(input_dim=39, output_dim=10, input_length=4)) 
# input_dim은 word_size 개수 + 1, oudtput_dim은 node 수, input_length은 문장의길이
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid')) # 이진분류
# model.summary() # 모델 시각화

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics='accuracy')
model.fit(pad_x, labels, epochs=100, batch_size=16)

#4. 평가, 예측
loss, acc = model.evaluate(pad_x, labels)
print('loss : ', loss)
print('acc : ', acc)
