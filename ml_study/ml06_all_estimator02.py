import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42, shuffle=True
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms : ', allAlgorithms)
print('몇 개??', len(allAlgorithms)) # 55

#3. 출력(평가, 예측)
for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률 : ', r2)
    except :
        print(name, '출력안됨')

# ARDRegression 의 정답률 :  0.5958111262169106
# AdaBoostRegressor 의 정답률 :  0.5015624845445779
# BaggingRegressor 의 정답률 :  0.7867745548675719
# BayesianRidge 의 정답률 :  0.5960121561109679
# CCA 출력안됨
# DecisionTreeRegressor 의 정답률 :  0.5909098098834507
# DummyRegressor 의 정답률 :  -4.0358175403820695e-06
# ElasticNet 의 정답률 :  -4.0358175403820695e-06
# ElasticNetCV 의 정답률 :  0.5974419516855731
# ExtraTreeRegressor 의 정답률 :  0.5441619390775975
# ExtraTreesRegressor 의 정답률 :  0.807296336334072
# GammaRegressor 의 정답률 :  0.01939908033622484
# GaussianProcessRegressor 의 정답률 :  -55.92324807215745
# GradientBoostingRegressor 의 정답률 :  0.7808803607143742
# HistGradientBoostingRegressor 의 정답률 :  0.8378448516934066
# HuberRegressor 의 정답률 :  0.5789831325144065
# IsotonicRegression 출력안됨
# KNeighborsRegressor 의 정답률 :  0.6940862656763869
# KernelRidge 의 정답률 :  0.5277900608154029
# Lars 의 정답률 :  0.5957681720649695
# LarsCV 의 정답률 :  0.5974630041137836
# Lasso 의 정답률 :  -4.0358175403820695e-06
# LassoCV 의 정답률 :  0.5983189073478872
# LassoLars 의 정답률 :  -4.0358175403820695e-06
# LassoLarsCV 의 정답률 :  0.5974630041137836
# LassoLarsIC 의 정답률 :  0.5957681720649695
# LinearRegression 의 정답률 :  0.5957681720649696
# LinearSVR 의 정답률 :  0.5806303717391312
# MLPRegressor 의 정답률 :  0.69786801649441
# MultiOutputRegressor 출력안됨
# MultiTaskElasticNet 출력안됨
# MultiTaskElasticNetCV 출력안됨
# MultiTaskLasso 출력안됨
# MultiTaskLassoCV 출력안됨
# NuSVR 의 정답률 :  0.659242208950624
# OrthogonalMatchingPursuit 의 정답률 :  0.47292621250963174
# OrthogonalMatchingPursuitCV 의 정답률 :  0.594936057215674
# PLSCanonical 출력안됨
# PLSRegression 의 정답률 :  0.5258951978240337
# PassiveAggressiveRegressor 의 정답률 :  -0.46425192685276784
# PoissonRegressor 의 정답률 :  0.039330685856654
