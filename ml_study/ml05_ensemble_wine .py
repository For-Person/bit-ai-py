import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 : [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=100, shuffle=True
)
# Scaler 적용
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = SVC()
# model = LinearSVC()
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result) 

# MinMaxScaler 적용 => 결과 acc :  0.9814814814814815
# StandardScaler 적용 => 결과 acc :  0.9629629629629629
# MaxAbsScaler 적용 => 결과 acc :  0.9814814814814815
# RobustScaler 적용 => 결과 acc :  0.9629629629629629

# 결과 acc :  0.7962962962962963 => Decision tree 적용
# 결과 acc :  0.9814814814814815 => Randomforest 적용