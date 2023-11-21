from sklearn.linear_model import LinearRegression
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd 
from sklearn.model_selection import train_test_split
import statsmodels.api

# 회귀분석 문제 5) 
# Kaggle 지원 dataset으로 회귀분석 모델(LinearRegression)을 작성하시오.
# testdata 폴더 : Consumo_cerveja.csv
# Beer Consumption - Sao Paulo : 브라질 상파울루 지역 대학생 그룹파티에서 맥주 소모량 dataset
# feature : Temperatura Media (C) : 평균 기온(C)
#              Precipitacao (mm) : 강수(mm)
# label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오
# 조건 : NaN이 있는 경우 삭제

data = pd.read_csv("../testdata/Consumo_cerveja.csv", usecols=[1,4,6])
data.dropna(inplace=True)

df = pd.DataFrame(data)
print(data.head(4))


x = data[['Temperatura Media (C)', 'Precipitacao (mm)']].apply(lambda x: pd.to_numeric(x.str.replace(',', '.'), errors='coerce'))
y = data['Consumo de cerveja (litros)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=12)
print(x)
print(y)

model = LinearRegression()
model.fit(x_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(x_test)

print('실제값 : ', y_test.values)
print('예측값 : ', np.round(y_pred,0))
print('결정계수 : ', r2_score(y_test, y_pred))

features = data[['Temperatura Media (C)', 'Precipitacao (mm)']].values
print(features[:5])

label =  data['Consumo de cerveja (litros)'].values
print(label[:5])


lmodel = LinearRegression().fit(features, label)
print('회귀계수(slope)', lmodel.coef_) 
print('회귀계수(intercept)', lmodel.intercept_)

pred = lmodel.predict(features)
print('예측값 : ', np.round(pred[:5], 1))  
print('실제값 :', label[:5])

# 모델 평가
print('MSE : ', mean_squared_error(label, pred))
print('r2_score : ', r2_score(label, pred))

# 예측
new_data = {'Temperatura Media (C)': [24], 'Precipitacao (mm)': [0]}
new_hp = pd.DataFrame(new_data)
new_pred = model.predict(new_hp)

print('%s강수량과 %s기온인 경우 예상 맥주소비량은 약 %s입니다.' % (new_data['Precipitacao (mm)'][0],new_data['Temperatura Media (C)'][0], new_pred[0]))









