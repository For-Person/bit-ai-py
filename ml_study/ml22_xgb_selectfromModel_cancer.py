import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 :  [0 1]

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

#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=20, 
          eval_set = [(x_train, y_train), (x_test, y_test)], 
          eval_metric='error')

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross validation
print('cv acc : ', score )

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)
# cv pred acc :  0.9473684210526315    

# SelectFromModel
from sklearn.feature_selection import SelectFromModel
thresholds = model.feature_importances_
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBClassifier()
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, ACC:%.2f%%" 
          %(thresh, select_x_train.shape[1], score*100))

# (455, 19) (114, 19)
# Thresh=0.006, n=19, ACC:96.49%
# (455, 7) (114, 7)
# Thresh=0.027, n=7, ACC:95.61%
# (455, 13) (114, 13)
# Thresh=0.011, n=13, ACC:95.61%
# (455, 8) (114, 8)
# Thresh=0.024, n=8, ACC:95.61%
# (455, 15) (114, 15)
# Thresh=0.010, n=15, ACC:95.61%
# (455, 17) (114, 17)
# Thresh=0.009, n=17, ACC:95.61%
# (455, 5) (114, 5)
# Thresh=0.043, n=5, ACC:96.49%
# (455, 1) (114, 1)
# Thresh=0.423, n=1, ACC:87.72%
# (455, 26) (114, 26)
# Thresh=0.003, n=26, ACC:95.61%
# (455, 24) (114, 24)
# Thresh=0.004, n=24, ACC:96.49%
# (455, 16) (114, 16)
# Thresh=0.010, n=16, ACC:95.61%
# (455, 25) (114, 25)
# Thresh=0.003, n=25, ACC:95.61%
# (455, 12) (114, 12)
# Thresh=0.014, n=12, ACC:95.61%
# (455, 20) (114, 20)
# Thresh=0.006, n=20, ACC:95.61%
# (455, 22) (114, 22)
# Thresh=0.005, n=22, ACC:95.61%
# (455, 18) (114, 18)
# Thresh=0.007, n=18, ACC:96.49%
# (455, 9) (114, 9)
# Thresh=0.022, n=9, ACC:95.61%
# (455, 14) (114, 14)
# Thresh=0.010, n=14, ACC:95.61%
# (455, 30) (114, 30)
# Thresh=0.000, n=30, ACC:95.61%
# (455, 27) (114, 27)
# Thresh=0.002, n=27, ACC:95.61%
# (455, 6) (114, 6)
# Thresh=0.042, n=6, ACC:96.49%
# (455, 11) (114, 11)
# Thresh=0.015, n=11, ACC:95.61%
# (455, 3) (114, 3)
# Thresh=0.074, n=3, ACC:94.74%
# (455, 4) (114, 4)
# Thresh=0.062, n=4, ACC:95.61%
# (455, 23) (114, 23)
# Thresh=0.005, n=23, ACC:96.49%
# (455, 28) (114, 28)
# Thresh=0.002, n=28, ACC:95.61%
# (455, 10) (114, 10)
# Thresh=0.018, n=10, ACC:95.61%
# (455, 2) (114, 2)
# Thresh=0.139, n=2, ACC:87.72%
# (455, 21) (114, 21)
# Thresh=0.005, n=21, ACC:96.49%
# (455, 30) (114, 30)
# Thresh=0.000, n=30, ACC:95.61%