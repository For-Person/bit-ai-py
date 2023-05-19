import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import time

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=42
)

#scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# param = {
#     'n_estimators' : [100],
#     'random_state' : [42, 62, 72],
#     'max_features' : [3, 4, 7]
# }

#2. 모델 (bagging)
xgb = XGBClassifier()
lgbm = LGBMClassifier()
catboost = CatBoostClassifier()
model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('catboost', catboost)],
    voting='soft', n_jobs=-1
    )
# model = GridSearchCV(voting, param, cv=kfold, refit=True, n_jobs=-1)

#3. 훈련
# start_time = time.time()
model.fit(x_train, y_train)
# end_time = time.time() - start_time

#4. 평가, 예측
# y_predict = model.predict(x_test)
# score = accuracy_score(y_test, y_predict)

# print('최적의 매개변수 : ', model.best_estimator_)
# print('최적의 파라미터 : ', model.best_params_)
# print('걸린시간 : ', end_time, '초')
# print('Voting 결과 : ', score)

classfiers = [catboost, xgb, lgbm]
for model in classfiers:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))

# CatBoostClassifier 정확도 :  1.0000
# XGBClassifier 정확도 :  1.0000
# LGBMClassifier 정확도 :  1.0000