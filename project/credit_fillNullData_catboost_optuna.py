import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import time

datasets = pd.read_csv('C:/ai_study/project/train.csv')
print(datasets.columns)
print(datasets.head(7))
data=datasets
# data = data.replace({np.nan: 'None'})
data = data.dropna(axis=0)
print(data)

data=data.replace({'Occupation' : 'Other'}, 0)
data=data.replace({'Occupation' : 'Salaried'}, 1)
data=data.replace({'Occupation' : 'Self_Employed'}, 2)
data=data.replace({'Occupation' : 'Entrepreneur'}, 3) 


data=data.replace({'Credit_Product' : 'No'}, 0)
data=data.replace({'Credit_Product' : 'Yes'}, 1)
# data=data.replace({'Credit_Product' : 'None'}, 2)

data=data.replace({'Gender' : 'Female'}, 0)
data=data.replace({'Gender' : 'Male'}, 1)

data=data.replace({'Is_Active' : 'No'}, 0)
data=data.replace({'Is_Active' : 'Yes'}, 1)

# # Avg_Account_Balance 값이 커서 로그로 바꿔주기
# data['Avg_Account_Balance'] = np.log1p(data['Avg_Account_Balance'])

data.info()
data.astype({'Credit_Product' : 'int'})

data['Region_Code']=data['Region_Code'].apply(lambda x: int(x.strip("RG")))
data['Channel_Code']=data['Channel_Code'].apply(lambda x: int(x.strip("X")))

data.drop('ID',axis = 1,inplace=True)  # 데이터 정제 (비어있는 값들 비워내고, 기존 문자열 데이터들 int형으로 변환)

x = pd.concat([data['Gender'],data['Age'],data['Region_Code'],data['Occupation'],data['Channel_Code'],data['Vintage'],data['Credit_Product'],data['Avg_Account_Balance'],data['Is_Active']], axis=1)
x = np.array(x)
y = np.array(data['Is_Lead'])
print(x)
print(type(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y,   # x,y 데이터
    train_size=0.8, # 훈련비율 
    random_state=42, #데이터를 난수값에 의해 추출한다는 의미 이며, 중요한 하이퍼파라미터임
    shuffle=True  # 데이터를 섞어서 가지고 올것인지를 정함
)

#kfold
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                        random_state=random_state)

# Scaler 적용 
from sklearn.preprocessing import StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

def objectiveCAT(trial: Trial, x_train, y_train, x_test):
    param = {'n_estimators' : trial.suggest_int('n_estimators', 500, 4000), 
         'depth' : trial.suggest_int('depth', 8, 16),
         'fold_permutation_block' : trial.suggest_int('fold_permutation_block', 1, 256),
         'learning_rate' : trial.suggest_float('learning_rate', 0, 1),
         'od_pval' : trial.suggest_float('od_pval', 0, 1),
         'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0, 4),
         'random_state' :trial.suggest_int('random_state', 1, 2000)}
    # 학습 모델 생성
    model = CatBoostClassifier(**param)
    CAT_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    # 모델 성능 확인
    score = accuracy_score(CAT_model.predict(x_test), y_test)
    return score
# MAE가 최소가 되는 방향으로 학습을 진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())

# n_trials 지정해주지 않으면, 무한 반복
study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 5)
print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))

# 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
print(optuna.visualization.plot_param_importances(study))

# 하이퍼파라미터 최적화 과정을 확인
optuna.visualization.plot_optimization_history(study)
plt.show()