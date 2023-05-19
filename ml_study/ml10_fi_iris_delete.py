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
# print(datasets.DESCR)

# drop_features
x = np.delete(x, 0, axis=1)
print(x.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, shuffle=True
)
# train_size 안에 validation도 포함되어있다.

#kfold
n_splits = 11
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
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross validation

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)
# cv pred acc :  0.9666666666666667

# ######## feature_importances_ ########
# print(model, " : ", model.feature_importances_)
# # RandomForestClassifier()  :  [0.11623676 0.03301891 0.41365352 0.43709082]

# # 시각화
# import matplotlib.pyplot as plt
# n_features = datasets.data.shape[1]
# plt.barh(range(n_features), model.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), datasets.feature_names)
# plt.title('iris Feature Importances')
# plt.ylabel('Feature')
# plt.xlabel('Importances')
# plt.ylim(-1, n_features)

# plt.show()