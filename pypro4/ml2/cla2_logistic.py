# Logistic  Regression
# weather dataset으로 다음날 비 여부를 예측하는 분류 모델 작성
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from dask.dataframe.shuffle import shuffle

data = pd.read_csv("../testdata/weather.csv")
print(data.head(2), data.shape)  # (366, 12)
data2 = pd.DataFrame()
data2 = data.drop(['Date','RainToday'], axis=1)
print(data2.head(2), data2.shape)  # (366, 10)
print(data2.columns)
print(data2.RainTomorrow.unique())   # ['Yes' 'No'] # dummy화 
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1,'No':0})
print(data2.RainTomorrow.unique())

# RainTomorrow : 종속(label), 나머지 변수는 : 독립(feature)
# 모델을 학습 후 검증하고 싶다면 train / test 로 분리
train, test = train_test_split(data2, test_size= 0.3, random_state=42, shuffle = True)
print(train.shape, test.shape)      # (256, 10) (110, 10)

# 분류모델
col_select = "+".join(train.columns.difference(['RainTomorrow']))
print(col_select)
myFormula = 'RainTomorrow ~ ' + col_select
# model = smf.glm(formula=myFormula, data=train, family=sm.families.Binomial()).fit()
model = smf.logit(formula=myFormula, data=train).fit()

print(model.summary())
print(model.params)
print('예측값 : ', np.rint(model.predict(test)[:5].values))       # 학습은 train  검증은 test
print('실제값 : ', test['RainTomorrow'][:5].values)

# 정확도 계산 
conf_mat = model.pred_table()   # glm 은 지원하지 않는다.
print('confusion matrix : \n', conf_mat)
print('정확도 : ', conf_mat[0][0] + conf_mat[1][1] / len(train))

from sklearn.metrics import accuracy_score
pred = model.predict(test)
print('정확도 :' , accuracy_score(test['RainTomorrow'], np.around(pred))) 


























