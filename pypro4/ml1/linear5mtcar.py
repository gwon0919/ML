# 단순 선형 회귀 : mtcars dataset, ols()
# 상관관계가 약한 경우와 강한 경우 분석 모델을 작성 후 비교

# 과학적 추론방식은 크게 두 가지로 분류
# 귀납법 : 개별 사례를 수집해서 일반적인 법칙을 생성 
# 연역법 : 사실이나 가정에 근거해 논리적 추론에 의해 결론을 도출

import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api
import numpy as np

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3), mtcars.shape)     # (32, 11)
print(mtcars.columns)
print(mtcars.describe())
print(mtcars.corr())
print(np.corrcoef(mtcars.hp, mtcars.mpg)[0, 1])     # -0.77616
print(np.corrcoef(mtcars.wt, mtcars.mpg)[0, 1])     # -0.86765

# 시각화 
# plt.scatter(mtcars.hp, mtcars.mpg)
# plt.xlabel('마력수')
# plt.ylabel('연비')
# slope, intercept = np.polyfit(mtcars.hp, mtcars.mpg, 1)
# plt.plot(mtcars.hp, mtcars.hp * slope + intercept, 'r')
# plt.show() 

print('\n단순  선형 회귀------------------- ')
result = smf.ols('mpg ~ hp', data=mtcars).fit()
print(result.summary())             # Prob (F-statistic):  1.79e-07 < 0.05 유의한 모델, Equared:  0.602
print('마력수:{}에 대한 연비 예측 결과:{}'.format(110, -0.0682 * 110 + 30.0989))
print('마력수:{}에 대한 연비 예측 결과:{}'.format(110, result.predict(pd.DataFrame({'hp': [110]}))))

print('\다중  선형 회귀------------------- ')
result2 = smf.ols('mpg ~ hp+wt', data=mtcars).fit()
print(result2.summary())          
print('마력수:{}, 차체무게:{} 에 대한 연비 예측 결과:{}'.format(110,5,  result2.predict(pd.DataFrame({'hp': [110], 'wt':[5]}))))
# 마력수:110, 차체무게:5 에 대한 연비 예측 결과:0    14.343092

# 회귀분석 문제 2) 
# testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
#   - 국어 점수를 입력하면 수학 점수 예측
#   - 국어, 영어 점수를 입력하면 수학 점수 예측

data = pd.read_csv('../testdata/student.csv')
print(data.head(3))
result3 = smf.ols('수학~ 국어', data=data).fit()
print(result3.summary())  
국어 = int(input('국어: '))
#print('국어:{}에 대한 수학 예측 결과:{}'.format(70, 0.5705 * 국어 + 32.1069))
newdf = pd.DataFrame({'국어':[국어]})
pred3 = result3.predict(newdf)
print('국어:{} 에 대한 수학 예측 점수 :{} ' , (newdf,pred3[0]))

result4 = smf.ols('수학 ~ 국어 + 영어', data=data).fit()
print(result4.summary())  

국어 = int(input('국어: '))
영어 = int(input('영어: '))
#print('국어:{}, 영어:{} 수학 예측 결과:{}'.format(70,80, 0.5705 * 국어 + 32.1069))
newdf2 = pd.DataFrame({'국어':[국어], '영어':[영어]})
pred4 = result4.predict(newdf2)
print('국어:{},영어:{} 점수로 예측한 수학점수는:{} '.format(newdf2['국어'][0],newdf2['영어'][0],pred4[0]))









