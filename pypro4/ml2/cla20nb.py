# Naive Bayes Classifier : weather dataset 사용 - 비가 올지 여부 분류

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("../testdata/weather.csv")
print(df.head(3), df.shape)  # (366, 12)
print(df.columns)


feature = df[['MinTemp', 'MaxTemp', 'Rainfall']] # feature
# print(df['RainTomorrow'][:10])
# label = df['RainTomorrow'].apply(lambda x:1 if x == 'Yes' else 0)
label = df['RainTomorrow'].map({'Yes':1, 'No': 0})
print(label[:10].values)

# train/test
train_x,test_x, train_y, test_y = train_test_split(feature, label, random_state=0, test_size=0.3)
print(train_x.shape,test_x.shape, train_y.shape, test_y.shape)  # (256, 3) (110, 3) (256,) (110,)

# model
gmodel = GaussianNB()
print(gmodel)
gmodel.fit(train_x,train_y)

pred = gmodel.predict(test_x)
print('예측값 : ', pred[:10])
print('실제값 : ', test_y[:10].values)

acc = sum(test_y == pred) / len(pred)
print('정확도 : ', acc)
print('분류 정확도 : ', accuracy_score(test_y, pred))
print('분류 보고서 : \n', classification_report(test_y,pred))

print('새 값으로 예측')
import numpy as np 
myWeather = np.array([[2,12,0],[22,43,50],[1,2,3]])
print('예측 결과 : ', gmodel.predict(myWeather))

# GaussianNB : 연속형 데이터
# BernoulliNB : 이전 데이터
# MultinomialNB : 텍스트 분류(카운트 데이터)















