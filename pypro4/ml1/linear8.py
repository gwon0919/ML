# LinearRegression  클래스를 사용해 선형회귀모델 작성 - 평가지표로 : MAE, MSE, RMSE, r2_score

from sklearn.linear_model import LinearRegression
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd 
from sklearn.model_selection import train_test_split

# 공부 시간에 따른 시험 점수표
df = pd.DataFrame({'studytime': [3,4,5,8,10,5,8,6,3,6,10,9,7,0,1,2],
                            'score': [76,74,74,89,92,75,84,82,73,81,89,88,83,40,70,69]})
print(df.head(2),df.shape)  # (16, 2) <-- 모집단 예시

# dataset을 분리해서 학습 및 검정을 실시 (모델의 과적합(overfitting) 방지 목적 중 하나)
train, test = train_test_split(df, test_size=0.4, random_state=12)  # 6:4로 분리
print(train)
print(test)
x_train = train[['studytime']]
# print(x_train, x_train.shape)  # (9, 1) 2차원(matrix)  형태로 작성. 
# 왜냐하면 sklearn의 분류 및 예측 클래스는 feature: 2차원, label : 1차원을 원함
y_train = train['score']    # 1차원 벡터
x_test = test[['studytime']]
y_test = test['score']
print(x_train.shape, x_test.shape, y_train.shape,y_test.shape) # (9, 1) (7, 1) (9,) (7,)

# LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)           # 모델 학습은 훈련용(train) 데이터로 
y_pred = model.predict(x_test)     # 모델 검정은 검정용(test) 데이터로
print('실제값 : ', y_test.values)
print('예측값 : ', np.round(y_pred,0))

# 결정 계수 수식 사용
# 잔차 구하기 
y_mean = np.mean(y_test)    # y의 평균
nomerator = np.sum(np.square(y_test - y_pred))  # SSE(오차 제곱합)
denomerator = np.sum(np.square(y_test - y_mean))  # SST(편차 제곱합)

r2 = 1 - nomerator / denomerator  # 1 - (SSE / SST)
print('결정계수 : ', r2)
print('결정계수 : ', r2_score(y_test, y_pred))  # 값이 동일하다. 결정계수 :  0.3629
# 결정계수는 분산을 기반으로 측정하므로 중심극한정리에 의해 표본 데이터가 많을수록 결정계수 수치도 높아진다.
# 무의미한 독립변수의 수가 늘면 결정계수 값이 늘어나는 경향이 있으므로 변수 선택에 주의가 필요하다. 결정계수 값은 맹신 불가~

import statsmodels.api
print('\n자동차 데이터(mtcars)로 선형모델 작성')
mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(2), mtcars.shape)  # (32, 11)
print(mtcars.corr(method='pearson'))    # 상관계수 확인

# mpg에 영향을 주는 feature로 hp로 선택 (단순선형회귀)
feature = mtcars[['hp']].values
print(feature[:5])
label = mtcars['mpg'].values
print(label[:5])

# plt.scatter(feature, label)
# plt.show()

lmodel = LinearRegression().fit(feature, label)
print('회귀계수(slope)', lmodel.coef_) # 기울기 
print('회귀계수(intercept)', lmodel.intercept_) # 절편

pred = lmodel.predict(feature)
print('예측값 : ', np.round(pred[:5], 1))  # 예측값 :  [22.6 22.6 23.8 22.6 18.2]
print('실제값 :', label[:5])                    # 실제값 : [21.  21.  22.8 21.4 18.7]

# 모델 성능 평가
print('MSE : ', mean_squared_error(label, pred))
print('r2_score : ', r2_score(label, pred))
print()
# 새로운  hp로 mpg를 예측
new_hp = [[110]]
new_pred = lmodel.predict(new_hp)
print('%s 마력인 경우 예상 연비는 약 %s입니다.'%(new_hp[0][0], new_pred[0]))
# 110 마력인 경우 예상 연비는 약 22.593749951750496입니다.























