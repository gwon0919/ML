# 회귀분석 문제 3)    
# kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음) Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.

import pandas as pd
import matplotlib.pyplot as plt
import scipy
from statsmodels.stats.outliers_influence import variance_inflation_factor
plt.rc('font', family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api
import numpy as np

data = pd.read_csv("../testdata/carseats.csv")
print(data.head(3),data.shape)
print(data.info())

# print(data.dtypes)

lm = smf.ols(formula='Sales ~ CompPrice + Income ', data=data).fit()
print(lm.summary())
# 1. 선형성 확인 (잔차 대 fitted values)
fitted = lm.predict(data)
residual = data['Sales'] - fitted
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
plt.title('선형성 확인')
plt.show()

# 2. 정규성 확인 (잔차의 정규분포)
residual_standardized = (residual - np.mean(residual)) / np.std(residual)
scipy.stats.probplot(residual_standardized, plot=plt)
plt.title('잔차의 정규성 확인')
plt.show()

# 3. 등분산성 확인 (잔차 대 fitted values의 표준편차)
sns.regplot(x=fitted, y=np.sqrt(np.abs(residual_standardized)), lowess=True, line_kws={'color':'red'})
plt.title('등분산성 확인')
plt.show()

# 4. 다중공선성 확인 (VIF)
X = data[['CompPrice', 'Income']]
vif = pd.DataFrame()
vif["변수"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
# print('---------')
# print(variance_inflation_factor(data.values, 1)) # tv,  OLS Regression Results에서 Intercept
# print(variance_inflation_factor(data.values, 2)) # radio
# vifdf = pd.DataFrame()
# vifdf['vif_value'] = [variance_inflation_factor(data.values, i) for i in range(1,3)]
# print(vifdf)
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lm).cooks_distance   # 극단값을 나타내는 지표를 반환
print(cd.sort_values(ascending=False).head())

# Sales 예측
pred_sales = lm.predict(data)
print(pred_sales)

# 모델 검증이 끝난 경우 모델을 저장
# 방법1
# import pickle
# with open('linear6m.model', 'wb') as obj:
#     pickle.dump(lm, obj)
#
# with open('linear6m.model', 'rb') as obj:
#     mymodel = pickle.load(obj)
    
# 방법2( 메모리 절약 가능)
import joblib
joblib.dump(lm,'linear.model')  
mymodel = joblib.load('linear.model')


    
    
    
    
    
    
    
    
    
    
    
    
