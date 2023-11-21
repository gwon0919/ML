# 단순 선형 회귀 분석 모델 작성 = ols () 함수 - OLS Regression Result  내용 알기
# 결정론적 선형회귀분석 방법 - 확률적 모형에 불확실이 덜하다.

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
plt.rc('font', family='malgun gothic')

df = pd.read_csv("../testdata/drinking_water.csv")
print(df.head(3), df.shape)
print(df.corr(method='pearson'))
# 독립변수(x, feature) : 적절성
# 종속변수(y, label) : 적절성
# 목적 : 주어진 feature와 결정적 기반에서 학습을 통해 최적의 회귀계수(slope, bias)를 찾아내는 것.
model = smf.ols(formula='만족도~ 적절성', data=df).fit()
# print(model.summary())
print(model.params)
print(model.pvalues)

# 예측값 
print(df.적절성[:5].values)
new_df = pd.DataFrame({'적절성': [4, 3, 4, 2, 2]})
new_pred = model.predict(new_df)
# 0.588 설명력이 있는 모델로 검정 
print('만족도 실제값 : ', df['만족도'][:5])
print('만족도 예측값 : ', new_pred)
