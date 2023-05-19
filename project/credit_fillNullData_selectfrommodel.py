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



feature_name = ['Gender','Age','Region_Code','Occupation','Channel_Code','Vintage','Credit_Product','Avg_Account_Balance','Is_Active']


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

# # parameters
# param = {
#     'learning_rate': [0.03, 0.05, 0.08, 0.1],
#     'depth': [4, 6, 8, 10], 
#     'l2_leaf_reg': [1, 3, 5, 7, 9],
#     'random_strength' : [42, 62, 72]
#     }

# 모델 구성
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier()
# model = CatBoostClassifier(depth=4, l2_leaf_reg=7, learning_rate=0.03, verbose=0)
model = LGBMClassifier(colsample_bylevel=0, colsample_bynode=0.1, 
                       colsample_bytree=0, feature_fraction=1.0, 
                       max_depth=5, min_data_in_leaf=100, 
                       n_estimators=200, num_leaves=10, 
                       reg_alpha=0.1, reg_lambda=0.1, verbose=0)

# model = GridSearchCV(catboost, param, refit=True, cv=kfold, verbose=1, n_jobs=-1)

# 컴파일, 훈련
import time
# start_time = time.time()
model.fit(x_train, y_train, early_stopping_rounds=20, 
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='merror')
# end_time = time.time() - start_time

#평가, 예측
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

result = model.score(x_test, y_test)

score = cross_val_score(model, x_train, y_train, cv=kfold) # cv : cross validation

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)

# SelectFromModel
from sklearn.feature_selection import SelectFromModel
thresholds = model.feature_importances_
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = LGBMClassifier(verbose=0)
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, ACC:%.2f%%" 
          %(thresh, select_x_train.shape[1], score*100))
    selected_feature_indices = selection.get_support(indices=True)
    selected_feature_names = [feature_name[i] for i in selected_feature_indices]
    print(selected_feature_names)

# (173120, 9) (43280, 9)
# Thresh=0.219, n=9, ACC:86.24%
# ['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# (173120, 3) (43280, 3)
# Thresh=16.808, n=3, ACC:85.92%
# ['Age', 'Occupation', 'Credit_Product']
# (173120, 8) (43280, 8)
# Thresh=0.735, n=8, ACC:86.21%
# ['Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# (173120, 1) (43280, 1)
# Thresh=34.618, n=1, ACC:84.85%
# ['Occupation']
# (173120, 5) (43280, 5)
# Thresh=3.836, n=5, ACC:86.00%
# ['Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product']
# (173120, 4) (43280, 4)
# Thresh=14.320, n=4, ACC:86.03%
# ['Age', 'Occupation', 'Vintage', 'Credit_Product']
# (173120, 2) (43280, 2)
# Thresh=26.277, n=2, ACC:84.85%
# ['Occupation', 'Credit_Product']
# (173120, 7) (43280, 7)
# Thresh=0.753, n=7, ACC:86.28%
# ['Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# (173120, 6) (43280, 6)
# Thresh=2.433, n=6, ACC:86.18%
# ['Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Is_Active'] 