# LinearRegression  클래스를 사용해 선형회귀모델 작성

from sklearn.linear_model import LinearRegression
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 편차가 큰 표본 데이터를 생성
sample_size = 100

np.random.seed(1)
x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 30
print(x[:10])
print(y[:10])
print('상관계수 : ', np.corrcoef(x,y))          # 0.9993

scaler = MinMaxScaler()         # 정규화
x_scaled = scaler.fit_transform(x.reshape(-1,1))
print(x_scaled[:10].flatten())  # flatten을 통해 2차원에서 1차원으로 축소 / 독립변수만 표준화 정규화 종속변수는 x

# 시각화 
# plt.scatter(x_scaled, y)
# plt.show()

model = LinearRegression().fit(x_scaled, y)
print(model)
y_pred = model.predict(x_scaled)
print('예측값 : ', y_pred[:10])
print('실제값 : ', y[:10])

# 모델 성능 확인 
# print(model.summary()) LinearRegression은 summary를 지원하지 않음 -> OLS사용해야함
def regScoreFunc(y_true, y_pred):
    print('r2_score(결정계수, 설명력):{} '.format(r2_score(y_true, y_pred)))
    print('explained_variance_score(설명 분산 점수):{} '.format(explained_variance_score(y_true, y_pred)))
    print('mean_squared_error(MSE, 평균 제곱근 오차):{} '.format(mean_squared_error(y_true, y_pred)))

regScoreFunc(y, y_pred)
# r2_score(결정계수, 설명력):0.9987875127274646 
# explained_variance_score(설명 분산 점수):0.9987875127274646     # 결정계수와 설명분산점수의 결과의 차이(편향)가 크다면 모델학습이 잘못됐다고 봄. ->모델학습이 잘못됐다 : 데이터 값이 문제가 있다. 
# mean_squared_error(MSE, 평균 제곱근 오차, SSE와 같음):86.14795101998747  # 숫자가 작을수록 좋다 = 실제값과 예측값의 차이가 작다 

print('------------------------------')
# 분산이 크게 다른 표본 데이터를 생성
x = np.random.normal(0,1,sample_size)
y = np.random.normal(0,500,sample_size) + x * 30
print(x[:10])
print(y[:10])
print('상관계수 : ', np.corrcoef(x,y))  # 0.00401

x_scaled2 = scaler.fit_transform(x.reshape(-1,1))
print(x_scaled2[:10].flatten()) 

model2 = LinearRegression().fit(x_scaled2, y)
y_pred2 = model2.predict(x_scaled2)
print('예측값 : ', y_pred2[:10])
print('실제값 : ', y[:10])

regScoreFunc(y, y_pred2)
# r2_score(결정계수, 설명력):1.6093526521765433e-05 
# explained_variance_score(설명 분산 점수):1.6093526521765433e-05 
# mean_squared_error(MSE, 평균 제곱근 오차):282457.9703485092 









