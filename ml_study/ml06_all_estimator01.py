import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42, shuffle=True
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
allAlgorithms = all_estimators(type_filter='classifier')
print('allAlgorithms : ', allAlgorithms)
print('몇 개??', len(allAlgorithms)) # 41

#3. 출력(평가, 예측)
for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except :
        print(name, '출력안됨')

# AdaBoostClassifier 의 정답률 :  1.0
# BaggingClassifier 의 정답률 :  1.0
# BernoulliNB 의 정답률 :  0.37777777777777777
# CalibratedClassifierCV 의 정답률 :  0.9111111111111111
# CategoricalNB 출력안됨
# ClassifierChain 출력안됨
# ComplementNB 의 정답률 :  0.7111111111111111
# DecisionTreeClassifier 의 정답률 :  1.0
# DummyClassifier 의 정답률 :  0.28888888888888886
# ExtraTreeClassifier 의 정답률 :  0.9555555555555556
# ExtraTreesClassifier 의 정답률 :  1.0
# GaussianNB 의 정답률 :  0.9777777777777777
# GaussianProcessClassifier 의 정답률 :  0.9111111111111111
# GradientBoostingClassifier 의 정답률 :  1.0
# HistGradientBoostingClassifier 의 정답률 :  1.0
# KNeighborsClassifier 의 정답률 :  1.0
# LabelPropagation 의 정답률 :  1.0
# LabelSpreading 의 정답률 :  1.0
# LinearDiscriminantAnalysis 의 정답률 :  1.0
# LinearSVC 의 정답률 :  0.9111111111111111
# LogisticRegression 의 정답률 :  0.9111111111111111
# LogisticRegressionCV 의 정답률 :  1.0
# MLPClassifier 의 정답률 :  0.9333333333333333
# MultiOutputClassifier 출력안됨
# MultinomialNB 의 정답률 :  0.9111111111111111
# NearestCentroid 의 정답률 :  0.9555555555555556
# NuSVC 의 정답률 :  1.0
# OneVsOneClassifier 출력안됨
# OneVsRestClassifier 출력안됨
# OutputCodeClassifier 출력안됨
# PassiveAggressiveClassifier 의 정답률 :  0.9555555555555556
# Perceptron 의 정답률 :  0.7333333333333333
# QuadraticDiscriminantAnalysis 의 정답률 :  1.0
# RadiusNeighborsClassifier 의 정답률 :  0.4444444444444444
# RandomForestClassifier 의 정답률 :  1.0
# RidgeClassifier 의 정답률 :  0.8444444444444444
# RidgeClassifierCV 의 정답률 :  0.8444444444444444
# SGDClassifier 의 정답률 :  0.9555555555555556
# SVC 의 정답률 :  1.0
# StackingClassifier 출력안됨
# VotingClassifier 출력안됨