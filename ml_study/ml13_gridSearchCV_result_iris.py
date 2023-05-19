import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 :  [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, shuffle=True
)
# train_size 안에 validation도 포함되어있다.

#kfold
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Scaler 적용 
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

param = [
    {'n_estimators' : [100, 200], 'max_depth':[6, 8, 10, 12], 
     'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]}
]

#2. 모델
from sklearn.model_selection import GridSearchCV
rf_model = RandomForestClassifier(max_depth=8, n_estimators=200, n_jobs=2)
model = GridSearchCV(rf_model, param, cv=kfold, verbose=1, 
                     refit=True, n_jobs=-1) # refit default 값 false라 true로 바꿔줘야함

#3. 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time
print('걸린시간 : ', end_time, '초')

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross validation
# print('cv acc : ', score )

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
# print('cv pred : ', y_predict) 


acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)
# 1번째 결과
# cv pred acc :  0.9666666666666667
# 2번째 결과
# cv pred acc :  0.9666666666666667 