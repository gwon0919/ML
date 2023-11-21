# 단순선형회귀분석 모델 생성
# 독립변수(연속형), 종속변수(연속형) : 두 변수는 상관관계가 있어야 하고, 나아가서는 인과관계가 있다는 가정하에 작성을 한다.
# 회귀 분석 각각의 데이터에 대한 잔차제곱의 합이 최소가 되는 수식을 도출해 내는 방법이다. 내부적으로 최소제곱법을 이용한다.

import statsmodels.api as sm
import numpy as np
from sklearn.datasets import make_regression
from astropy.units import yyr
from statsmodels.treatment.tests.test_teffects import formula

np.random.seed(12)

# 모델 생성 맛보기 
print('방법1 :  make_regression 사용. 모델 생성 안됨')
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
print('x, y, coef : ',x,y,coef)         # 회귀식 y= wx + b        
# 랜덤하게 x값을 생성  [[-1.70073563]  [-0.67794537] [ 0.31866529] ...
# 생성된 x값에 대한 y값(종속변수)을 제시     -52.17214291   39.34130801  128.51235594 ...
# 학습 후 수식이 완성(모델)  y = wx + b  ==> 예측값 y =  89.47430739278907 * x + 100
pred_Y = 89.47430739278907 * -1.70073563 + 100      # 작성된 모델로 x에 대한 예측값 y를 출력
print('y의 실제값은 ' ,  -52.17214291)
print('x값 -1.70073563에 대한 예측값 y는 ', pred_Y)     # -52.17214255248879

new_pred_Y = 89.47430739278907 * 1234.5678 + 100
print('내가 궁금한 1234.5678에 대한 예측값은 ', new_pred_Y)     # 110562.09883443934

xx = x
yy = y
print('방법 2 : LinearRegression을 사용. 모델 생성 됨')
from sklearn.linear_model import LinearRegression
model = LinearRegression()
fit_model = model.fit(xx,yy)
print('기울기(slope) : ' ,fit_model.coef_)          # [89.47430739]
print('절편(bias) : ', fit_model.intercept_)      # 100.0

# 예측값 확인 
y_new = fit_model.predict(xx[[0]])
print('y_new : ', y_new)                               # [-52.17214291]
1234.56789
y_new2 = fit_model.predict(xx[[12]])
print('정말 궁금한 새로운 x값에 대한 예측 결과 y는', y_new2)     # [7.88349776]
y_new2 = fit_model.predict(xx[[33]])
print('정말 궁금한 새로운 x값에 대한 예측 결과 y는', y_new2)     # [208.24569942]

print('방법 3 : ols 사용. 모델 생성 됨')
import statsmodels.formula.api as smf
import pandas as pd 
print(xx.shape)     # (50, 1)
x1 = xx.flatten()    # 차원 축소
print(x1.shape)
y1 = yy 
print(y1.shape)

data = np.array([x1,y1])
df = pd.DataFrame(data.T)
df.columns = ['x1','y1']
print(df.head(3), len(df))
model2 = smf.ols(formula=' y1 ~ x1', data= df).fit()
# model2.fit()
print(model2.summary())
#예측값 확인
print(x1[:2])   # [-1.70073563 -0.67794537]
new_df = pd.DataFrame({'x1':[-1.70073563, -0.67794537]})
print(new_df)
new_pred = model2.predict(new_df)
print('예측값 new_pred :', new_pred)
print('실제값 : ', df.y1[:2])

# 전혀 새로운 독립변수(x) 값에 예측값 확인
new2_df = pd.DataFrame({'x1':[111, -6.12345]})
new2_pred = model2.predict(new2_df)
print('예측값 new2_pred :', new2_pred)


















