import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(77) # weight 난수값 조정

#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 :  [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, shuffle=True
)
# train_size 안에 validation도 포함되어있다.

#kfold
n_splits = 11
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Scaler 적용 
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
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
    model = CatBoostRegressor(**param)
    CAT_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    # 모델 성능 확인
    score = r2_score(CAT_model.predict(x_test), y_test)
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

# [I 2023-05-17 15:21:28,404] Trial 4 finished with value: -57049.030638860364 and parameters: {'n_estimators': 2356, 'depth': 16, 'fold_permutation_block': 149, 'learning_rate': 0.17019066533220262, 'od_pval': 0.34470488861338755, 'l2_leaf_reg': 3.9277536624376186, 'random_state': 195}. Best is trial 0 with value: -2993.36931017284.
# Best trial : score -2993.36931017284, 
# params {'n_estimators': 2596, 'depth': 12, 'fold_permutation_block': 142, 'learning_rate': 0.5224915428781957, 'od_pval': 0.571161341604556, 'l2_leaf_reg': 0.6973445597722256, 'random_state': 785}
# Figure({
#     'data': [{'cliponaxis': False,
#               'hovertemplate': [learning_rate (FloatDistribution):
#                                 0.05060541343764662<extra></extra>, depth
#                                 (IntDistribution):
#                                 0.09382831686295402<extra></extra>, od_pval        
#                                 (FloatDistribution):
#                                 0.09879556890253281<extra></extra>,
#                                 fold_permutation_block (IntDistribution):
#                                 0.1023071549741552<extra></extra>, n_estimators    
#                                 (IntDistribution):
#                                 0.10510857567765701<extra></extra>, l2_leaf_reg    
#                                 (FloatDistribution):
#                                 0.2699612884182084<extra></extra>, random_state    
#                                 (IntDistribution):
#                                 0.279393681726846<extra></extra>],
#               'marker': {'color': 'rgb(66,146,198)'},
#               'orientation': 'h',
#               'text': [0.05, 0.09, 0.10, 0.10, 0.11, 0.27, 0.28],
#               'textposition': 'outside',
#               'type': 'bar',
#               'x': [0.05060541343764662, 0.09382831686295402, 0.09879556890253281, 
#                     0.1023071549741552, 0.10510857567765701, 0.2699612884182084,   
#                     0.279393681726846],
#               'y': [learning_rate, depth, od_pval, fold_permutation_block,
#                     n_estimators, l2_leaf_reg, random_state]}],
#     'layout': {'showlegend': False,
#                'template': '...',
#                'title': {'text': 'Hyperparameter Importances'},
#                'xaxis': {'title': {'text': 'Importance for Objective Value'}},     
#                'yaxis': {'title': {'text': 'Hyperparameter'}}}
# })