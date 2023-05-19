import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import re

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer
#boost모델 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
#scale 적용
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# stratifiedKFold 적용
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import time
# 데이터 정제 전략 
# 계좌 금액의 값범위가 넓으므로 log값취하여 범위 축소
# age와 region은 다음과 같은 기준으로바꿔보자
# age와 target을 비교해 봤을 때, 상위로부터 구룹군 3~5개로 묶어서 분류
# 위처럼 실행 후, 안한 값과 비교해봤을때 auc가 떨어지면 원복 
# https://www.kaggle.com/code/shivamsingh96/credit-card-lead-prediction-auc-score-90 
# 해당 url 내용 통해서 그래프 및 데이터정제
greedSearch = False
train = pd.read_csv('C:/ai_study/project/train.csv')
test = pd.read_csv('C:/ai_study/project/test.csv')

train=train.dropna(axis=0)
test=test.dropna(axis=0)

Target = pd.DataFrame(train['Is_Lead'])

train = train.drop(['Is_Lead', 'ID'], axis = 1)
test = test.drop(['ID'], axis = 1)



data = pd.concat([train, test])
print(data.shape)

data['Avg_Account_Balance'] = np.log(data['Avg_Account_Balance'])

data_num_cols = data._get_numeric_data().columns 
data_cat_cols = data.columns.difference(data_num_cols)

print(data_num_cols)
print('-------------------------')
print(data_cat_cols)

data_num_data = data.loc[:, data_num_cols]
data_cat_data = data.loc[:, data_cat_cols]
print("Shape of num data:", data_num_data.shape)
print("Shape of cat data:", data_cat_data.shape)

s_scaler = StandardScaler()
data_num_data_s = s_scaler.fit_transform(data_num_data)

data_num_data_s = pd.DataFrame(data_num_data_s, columns = data_num_cols)

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
data_cat_data = data_cat_data.apply(LabelEncoder().fit_transform)


data_num_data_s.reset_index(drop=True, inplace=True)
data_cat_data.reset_index(drop=True, inplace=True)
#df = pd.concat([df1, df2], axis=1)
data_new = pd.concat([data_num_data_s, data_cat_data], axis = 1)


train_new = data_new.iloc[:216400,]
test_new = data_new.iloc[216401:,]

print("Shape of train data:", train_new.shape)
print("Shape of test data:", test_new.shape)


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(train_new,Target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)

param = {
    'learning_rate': [0.1,0.15,0.175,0.2],
        'depth': [4],
        'l2_leaf_reg': [0.2,0.4,0.6,0.8,1]
}
from sklearn.model_selection import GridSearchCV
# 모델링 및 훈련
if greedSearch == True:
    cat = CatBoostClassifier()
    model = GridSearchCV(cat,param, verbose= 1, refit=True, n_jobs=-1)
    print('최적의 파라미터 : ',model.best_params_)
    print('최적의 매개변수 : ', model.best_estimator_)
    print('best_score: ',model.best_score_)
    print('model_score : ', model.score(x_test, y_test))
else:
    model = CatBoostClassifier()#최적의 파라미터 : {'depth': 4, 'l2_leaf_reg': 0.2, 'learning_rate': 0.2}
    model.fit(x_train,y_train)
    model.fit(x_train,y_train)

    result = model.score(x_test, y_test)
    print('결과 acc : ', result )

    score = cross_val_score(model,
                        x_test, y_test) #cv: cross_validation
    print('cross validation acc : ', score)

    y_predict = cross_val_predict(model, x_test,y_test)

    print('cv predict : ',y_predict)

    acc = accuracy_score(y_test, y_predict)
    print('cv pred acc : ', acc)
    