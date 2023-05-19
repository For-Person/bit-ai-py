import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 :  [0 1 2]

#kfold
n_splits = 5
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Scaler 적용 (?)
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)

#2. 모델
# model = SVC()
model = RandomForestClassifier()

#3. 훈련
model.fit(x, y)

#4. 평가, 예측
result = model.score(x, y)
print('결과 acc : ', result) 

# kfold 결과 acc : 1.0 => Randomforest 적용