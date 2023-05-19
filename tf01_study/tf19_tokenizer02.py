import numpy as np
from keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 매우 매우 매우 매우 맛있는 밥을 엄청 많이 많이 많이 먹어서 매우 배가 부르다'
text2 = '나는 딥러닝이 정말 재미있다. 재미있어하는 내가 너무 너무 너무 너무 멋있다. 또 또 또 얘기해봐'

token = Tokenizer()
token.fit_on_texts([text1, text2]) # fit on 하면서 index 생성
# index = token.word_index
print(token.word_index)

# {'매우': 1, '너무': 2, '많이': 3, '또': 4, '나는': 5, '진짜': 6, '맛있는': 7, '밥을': 8, 
# '엄청': 9, '먹어서': 10, '배가': 11, '부르다': 12, '딥러닝이': 13, '정말': 14, '재미있다': 15, 
# '재미있어하는': 16, '내가': 17, '멋있다': 18, '얘기해봐': 19}

x = token.texts_to_sequences([text1, text2])
print(x) 
# [[5, 6, 1, 1, 1, 1, 7, 8, 9, 3, 3, 3, 10, 1, 11, 12], # text1
# [5, 13, 14, 15, 16, 17, 2, 2, 2, 2, 18, 4, 4, 4, 19]] # text2

from keras.utils import to_categorical

x_new = x[0] + x[1]
print(x_new)
# [5, 6, 1, 1, 1, 1, 7, 8, 9, 3, 3, 3, 10, 1, 11, 12, 5, 13, 14, 15, 16, 17, 2, 2, 2, 2, 18, 4, 4, 4, 19]

# x = to_categorical(x_new) # onehotencoding 하면 index수+1개로 만들어짐
# print(x)
# print(x.shape) # (31, 20)

####### OneHotEncoder 수정 #########
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
x = np.array(x_new)
print(x.shape) # (31,)
x = x.reshape(-1, 1)
print(x.shape) # (31, 1)
# x = x.reshape(-1, 11, 9)
onehot_encoder.fit(x)
x = onehot_encoder.transform(x)
print(x)
print(x.shape) # (31, 19)
# # AttributeError: 'list' object has no attribute 'reshape'
