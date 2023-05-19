import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time

#1. 데이터
datasets = fetch_california_housing()
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
xgb = XGBRegressor(max_features=7, n_estimators=100, random_state=42)
lgbm = LGBMRegressor(max_features=7, n_estimators=100, random_state=42)
catboost = CatBoostRegressor(n_estimators=100, random_state=42)
model = VotingRegressor(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('catboost', catboost)],
    n_jobs=-1
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

regressors = [catboost, xgb, lgbm]
for model in regressors:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = r2_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4f}'.format(class_names, score))

# CatBoostRegressor 정확도 :  0.8492
# XGBRegressor 정확도 :  0.8287
# LGBMRegressor 정확도 :  0.8365