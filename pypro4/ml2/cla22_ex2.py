import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from xgboost.core import DMatrix

df = pd.read_csv('../testdata/bike_dataset.csv')
print(df.head(3), df.shape)
print(df.columns)

# 'datetime' 열 제거
df = df.drop('datetime', axis=1)

categorical_columns = ['weather']

# LabelEncoder를 사용하여 범주형 열을 숫자로 변환
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

features = df.drop('count', axis=1)
label = df['count']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=12)

# DMatrix 생성 시 enable_categorical=True로 설정
dtrain = DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = DMatrix(X_test, label=y_test, enable_categorical=True)

# XGBRegressor 사용
model = XGBRegressor()
model.fit(X_train, y_train)

# 특성 중요도 시각화
plot_importance(model)
plt.show()

features2 = features[['registered', 'casual', 'humidity']].copy()


# 학습 및 테스트 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(features2, label, test_size=0.2, random_state=6413)

# GaussianNB 모델 생성 및 학습
gmodel = GaussianNB()
gmodel.fit(train_x, train_y)

# 예측
pred = gmodel.predict(test_x)
print('예측값 : ', pred[:10])
print('실제값 : ', test_y[:10])

# 정확도 평가
acc = sum(test_y == pred) / len(pred)
print('정확도 : ', acc) # 0.1042
print('정확도 : ', accuracy_score(test_y, pred)) # 0.1042
print('분류 보고서 : \n', classification_report(test_y, pred))









