print('방법 4 : linregress 사용. 모델 생성 됨')

from scipy import stats
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# IQ에 따른 시험 점수 값 예측
score_iq = pd.read_csv("../testdata/score_iq.csv")
print(score_iq.head(3), score_iq.shape)
x = score_iq.iq
y = score_iq.score

# 상관계수 확인
print(np.corrcoef(x, y)[0,1])
print(score_iq.corr())          # 0.882220 
#plt.scatter(x,y)
#plt.show()

# 인과관계가 있다는 가정하에 선형회귀분석 모델 생성
model = stats.linregress(x, y)
print(model)
print('x 기울기 : ', model.slope)
print('y 절편 : ', model.intercept)
print('상관계수 : ', model.rvalue)
print(' p값 : ', model.pvalue)

# plt.scatter(x,y)
# plt.plot(x, model.slope * x + model.intercept, c='r')
# plt.show()
# 회귀모델 수식 : y = model.slope * x + model.intercept
print('점수 예측 : ', model.slope * 80 + model.intercept)
print('점수 예측 : ', model.slope * 120 + model.intercept)
print('점수 예측 : ', model.slope * 140 + model.intercept)
# predict() 지원 x : numpy의 ployval([기울기, 절편], x)을 사용
print('점수 실제 값 : ', score_iq['score'][:5].values)
print('점수 예측 : ', np.polyval([model.slope, model.intercept], np.array(score_iq['iq'][:5])))

new_df = pd.DataFrame({'iq': [83, 90, 100, 127, 141]})
print('뉴 점수 예측 : ', np.polyval([model.slope, model.intercept], new_df).flatten)











