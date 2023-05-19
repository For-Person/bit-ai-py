import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
import re

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
import time
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
#
#
def data_refine(dataset):
    predict_data = dataset
    predict_data=predict_data.replace({'Occupation' : 'Other'}, 0)
    predict_data=predict_data.replace({'Occupation' : 'Salaried'}, 1)
    predict_data=predict_data.replace({'Occupation' : 'Self_Employed'}, 2)
    predict_data=predict_data.replace({'Occupation' : 'Entrepreneur'}, 3) 

    predict_data=predict_data.replace({'Credit_Product' : 'No'}, 0)
    predict_data=predict_data.replace({'Credit_Product' : 'Yes'}, 1)

    predict_data=predict_data.replace({'Gender' : 'Female'}, 0)
    predict_data=predict_data.replace({'Gender' : 'Male'}, 1)

    predict_data=predict_data.replace({'Is_Active' : 'No'}, 0)
    predict_data=predict_data.replace({'Is_Active' : 'Yes'}, 1)
    predict_data['Region_Code']=predict_data['Region_Code'].apply(lambda x: int(x.strip("RG")))
    predict_data['Channel_Code']=predict_data['Channel_Code'].apply(lambda x: int(x.strip("X")))
    predict_x = predict_data
    predict_y = predict_data['Credit_Product']
    return[predict_x, predict_y]

#기본 데이터
datasets = pd.read_csv('C:/Users/bitcamp/Downloads/project-20230509T112356Z-001/project/credit_card_prediction/train.csv')
datasets.drop('ID',axis = 1,inplace=True) 
data = datasets

#print('===========================')
target_data = data[data['Credit_Product'].isnull()]
data = target_data.replace({np.nan: 1})

predict_xData, trash_data = data_refine(target_data)
predict_xData.drop('Credit_Product',axis = 1,inplace=True) 

#print(len(predict_xData))
#print('빈칸인 데이터==============================================================================')



#빈 y값 예측하기

#비어있는 행 제거
datasets[datasets['Credit_Product'].isnull()]
datasets['Credit_Product'].isnull().sum()
datasets.isnull().sum()
data=datasets.dropna(axis=0)
#print(data.shape)
x, y = data_refine(data)
real_x = x.copy()
#print(real_x.shape)
x.drop('Credit_Product',axis = 1,inplace=True) 
print('y값----------------------')
print(y)   
print('x값----------------------')
print(x)
y = y.transpose()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8,random_state=42,shuffle=True
)

#KFold
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)

#Scaler 적용
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
#model = SVC()
#model = LinearSVC()
model = CatBoostClassifier()

#3.  훈련
model.fit(x_train,y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result )

score = cross_val_score(model,
                        x_test, y_test,
                        cv= kfold) #cv: cross_validation
print('cross validation acc : ', score)

y_predict = cross_val_predict(model, x_test,y_test,cv = kfold)

print('cv predict : ',y_predict)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)

predict_y = model.predict(predict_xData)
print(predict_y)
print('===================================================================')
print(sum(predict_y) / predict_xData.__len__())
print('===================================================================')



#####이제 빈값을 구했으니 본래데이터에 예측값 합치기
#print(type(predict_xData['Credit_Product']),type(predict_y['Credit_Product']))

predict_y = pd.DataFrame(predict_y)
#print('111')
#print(target_data.shape)
#print(real_x.shape)
target_data['Credit_Product'].map(predict_y.all())


predict_y.info()
#print(len(predict_y))
#print(target_data['Credit_Product'].sum() / target_data['Credit_Product'].__len__())
#print(predict_y.sum() / predict_y.__len__())


###이제 본래 작업

#1. 데이터
#print(x.shape)
#print(real_x.shape)
print(target_data)
print('1212122121')
target_data = pd.DataFrame(target_data)
target_data.info()
target_data, trash_data = data_refine(target_data)
print(target_data.shape)
print('===================================')
x2 = pd.concat([real_x,target_data],axis = 0)
x2.info()
print(x2.shape)
print(x2)
y2 = x2['Is_Lead']
x2.drop('Is_Lead',axis = 1,inplace=True) 

#y = y.transpose()
x2 = np.array(x2)
y2 = np.array(y2)

x_train2, x_test2, y_train2, y_test2 = train_test_split(
    x2, y2, train_size=0.8,random_state=42,shuffle=True
)

#KFold
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)

#Scaler 적용
scaler = MinMaxScaler()
print(x_train2)
print(x_train2.shape)
scaler.fit(x_train2)
x_train2 = scaler.transform(x_train2)
x_test2 = scaler.transform(x_test2)

#2. 모델
#model = SVC()
#model = LinearSVC()
model = CatBoostClassifier()

#3.  훈련
model.fit(x_train2,y_train2)


#4. 평가, 예측
result = model.score(x_test2, y_test2)
print('결과 acc : ', result )

score = cross_val_score(model,
                        x_test2, y_test2,
                        cv= kfold) #cv: cross_validation
print('cross validation acc : ', score)

y_predict = cross_val_predict(model, x_test2,y_test2,cv = kfold)

print('cv predict : ',y_predict)

acc = accuracy_score(y_test2, y_predict)
print('cv pred acc : ', acc)


##### feature importance ####
'''print(model, " : ", model.feature_importances_)

#시각화
import matplotlib.pyplot as plt
n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title('Iris Feature_importance')
plt.ylabel('Feature')
plt.xlabel('importance')
plt.ylim(-1, n_features)
plt.show()'''
