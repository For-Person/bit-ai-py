# [실습] svm모델과 나의 tf keras 모델 성능 비교하기
# 1. iris
# 2. cancer
# 3. wine
# 4. california svr
import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (20640, 8) (20640,)
print('y의 라벨값 : ', np.unique(y)) 
# y의 라벨값 : [0.14999 0.175   0.225   ... 4.991   5.      5.00001]

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=100, shuffle=True
# )

#2. 모델
# model = SVR()
model = LinearSVR()

#3. 훈련
model.fit(x, y)

#4. 평가, 예측
result = model.score(x, y)
print('결과 r2 : ', result) 

# 결과 acc :  -0.01658668690926901 => SVR 모델
# 결과 acc :  -0.48801425901257156 => LinearSVR 모델
# 결과 acc :  -1.8851681778185392 => LinearSVR 모델(훈련 테스트 데이터셋 했을때)
# california는 svm 모델 안쓴게 제일 나음