import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
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

#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=20, 
          eval_set = [(x_train, y_train), (x_test, y_test)], 
          eval_metric='merror')

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross validation

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)
# cv pred acc :  0.9444444444444444

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
    
# (142, 8) (36, 8)
# Thresh=0.015, n=8, ACC:97.22%
# (142, 6) (36, 6)
# Thresh=0.019, n=6, ACC:94.44%
# (142, 9) (36, 9)
# Thresh=0.012, n=9, ACC:97.22%
# (142, 11) (36, 11)
# Thresh=0.004, n=11, ACC:97.22%
# (142, 5) (36, 5)
# Thresh=0.029, n=5, ACC:94.44%
# (142, 7) (36, 7)
# Thresh=0.016, n=7, ACC:94.44%
# (142, 4) (36, 4)
# Thresh=0.113, n=4, ACC:94.44%
# (142, 13) (36, 13)
# Thresh=0.000, n=13, ACC:97.22%
# (142, 13) (36, 13)
# Thresh=0.000, n=13, ACC:97.22%
# (142, 2) (36, 2)
# Thresh=0.189, n=2, ACC:88.89%
# (142, 10) (36, 10)
# Thresh=0.010, n=10, ACC:97.22%
# (142, 1) (36, 1)
# Thresh=0.437, n=1, ACC:61.11%
# (142, 3) (36, 3)
# Thresh=0.156, n=3, ACC:91.67%