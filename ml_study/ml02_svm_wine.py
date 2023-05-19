# [실습] svm모델과 나의 tf keras 모델 성능 비교하기
# 1. iris
# 2. cancer
# 3. wine
# 4. california svr
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 : [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)

#2. 모델
# model = SVC()
model = LinearSVC()

#3. 훈련
model.fit(x, y)

#4. 평가, 예측
result = model.score(x, y)
print('결과 acc : ', result) 

# 결과 acc :  0.7078651685393258 => SVR 모델
# 결과 acc :  0.9157303370786517 => LinearSVC 모델
# 결과 acc :  0.797752808988764 => LinearSVC 모델(훈련 테스트 데이터셋 했을때)
# wine는 LinearSVC 모델 쓴게 제일 나음