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

# Avg_Account_Balance 값이 커서 로그로 바꿔주기
data['Avg_Account_Balance'] = np.log1p(data['Avg_Account_Balance'])

data.loc[(data['Age'] < 40), 'Age'] = 0
data.loc[(data['Age'] >= 40) & (data['Age'] < 60), 'Age'] = 1
data.loc[(data['Age'] >= 60), 'Age'] = 2

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
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
# scaler = MinMaxScaler()     
scaler = StandardScaler() 
# scaler = MaxAbsScaler()   
# scaler = RobustScaler() 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델 구성
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

# model = XGBClassifier()
model = CatBoostClassifier(depth=4, l2_leaf_reg=7, learning_rate=0.03)

# 컴파일, 훈련

model.fit(x_train, y_train)


#평가, 예측
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

result = model.score(x_test, y_test)

score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross validation

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)