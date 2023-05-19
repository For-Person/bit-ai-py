import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
import time

#1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
x = datasets.data   
y = datasets.target 
print(x.shape, y.shape) #(569, 30), (569, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=100, shuffle=True
)

#2. 모델구성
# model = Sequential()
# model.add(Dense(100, input_dim=30))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(50))
# model.add(Dense(1, activation='sigmoid'))  # 이진분류는 무조건 아웃풋 레이어의 활성화함수를 'sigmoid'로 해줘야한다.

# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
# ## earlyStopping
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min',
#                               verbose=1, restore_best_weights=True)
# # Model Check point
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
#                       save_best_only=True, 
#                       filepath='./_mcp/tf18_cancer.hdf5')
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=200, validation_split=0.2, 
#                  callbacks=[earlyStopping, mcp], verbose=1)
# end_time = time.time() - start_time
model = load_model('./_mcp/tf18_cancer.hdf5')

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
loss, acc, mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# [실습] accuracy_score를 출력하기
# y_predict 반올림하기
# y_predict = np.where(y_predict > 0.5, 1, 0)
y_predict = np.round(y_predict)
# for i in range(y_predict.size): # 반복문이용해서 반올림하기
#     if y_predict[i] >= 0.5:
#         y_predict[i] = 1
#     else:
#         y_predict[i] = 0 
acc = accuracy_score(y_test, y_predict)
print('loss : ', loss)
print('y_predict : ', y_predict)
print('accuracy_score : ', acc)
print('mse : ', mse) # 0.12175898998975754
# print('걸린시간 : ', end_time) # 1.0686357021331787
# loss : [0.4436775743961334, 0.0992577075958252]
# accuracy_score : 0.847953216374269 # np.where() 사용
# accuracy_score : 0.8713450292397661 # np.round() 사용
