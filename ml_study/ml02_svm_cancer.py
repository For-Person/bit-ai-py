# [실습] svm모델과 나의 tf keras 모델 성능 비교하기
# 1. iris
# 2. cancer
# 3. wine
# 4. california svr
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 :  [0 1]

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=100, shuffle=True
# )

#2. 모델
model = SVC()
# model = LinearSVC()

#3. 훈련
model.fit(x, y)

#4. 평가, 예측
result = model.score(x, y)
print('결과 acc : ', result) 

# 결과 acc :  0.9226713532513181 => SVC 모델
# 결과 acc :  0.8980667838312829 => LinearSVC 모델
# 결과 acc :  0.9239766081871345 => LinearSVC 모델(훈련 테스트 데이터셋 했을때)
# cancer는 svc 모델이 더 나음