import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_wine()
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

# AdaBoostClassifier 의 정답률 :  0.9259259259259259
# BaggingClassifier 의 정답률 :  0.9629629629629629
# BernoulliNB 의 정답률 :  0.3888888888888889
# CalibratedClassifierCV 의 정답률 :  0.9814814814814815
# CategoricalNB 의 정답률 :  0.4444444444444444
# ClassifierChain 출력안됨
# ComplementNB 의 정답률 :  0.8888888888888888
# DecisionTreeClassifier 의 정답률 :  0.9444444444444444
# DummyClassifier 의 정답률 :  0.3888888888888889
# ExtraTreeClassifier 의 정답률 :  0.9444444444444444
# ExtraTreesClassifier 의 정답률 :  1.0
# GaussianNB 의 정답률 :  1.0
# GaussianProcessClassifier 의 정답률 :  1.0
# GradientBoostingClassifier 의 정답률 :  0.9074074074074074
# HistGradientBoostingClassifier 의 정답률 :  1.0
# KNeighborsClassifier 의 정답률 :  0.9444444444444444
# LabelPropagation 의 정답률 :  0.9629629629629629
# LabelSpreading 의 정답률 :  0.9629629629629629
# LinearDiscriminantAnalysis 의 정답률 :  1.0
# LinearSVC 의 정답률 :  0.9814814814814815
# LogisticRegression 의 정답률 :  1.0
# LogisticRegressionCV 의 정답률 :  0.9814814814814815
# MLPClassifier 의 정답률 :  0.9814814814814815
# MultiOutputClassifier 출력안됨
# MultinomialNB 의 정답률 :  0.9629629629629629
# NearestCentroid 의 정답률 :  0.9814814814814815
# NuSVC 의 정답률 :  0.9814814814814815
# OneVsOneClassifier 출력안됨
# OneVsRestClassifier 출력안됨
# OutputCodeClassifier 출력안됨
# PassiveAggressiveClassifier 의 정답률 :  0.9814814814814815
# Perceptron 의 정답률 :  0.9814814814814815
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9814814814814815
# RadiusNeighborsClassifier 의 정답률 :  0.9444444444444444
# RandomForestClassifier 의 정답률 :  1.0
# RidgeClassifier 의 정답률 :  0.9814814814814815
# RidgeClassifierCV 의 정답률 :  0.9814814814814815
# SGDClassifier 의 정답률 :  0.9814814814814815
# SVC 의 정답률 :  0.9814814814814815
# StackingClassifier 출력안됨
# VotingClassifier 출력안됨