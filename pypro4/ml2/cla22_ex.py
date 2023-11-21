import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('../testdata/mushrooms.csv')

# class를 제외한 독립변수 선택
features = df.drop('class', axis=1)
features = pd.get_dummies(features)
label = df['class'].map({'p':1, 'e':0})


# 학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=12)

# XGBClassifier 모델 생성 및 학습
model = XGBClassifier()
model.fit(X_train, y_train)

# 특성 중요도 시각화
plot_importance(model)
plt.show()
print(df.columns)

features2 = features[['gill-size_b', 'odor_n', 'bruises_f']].copy()


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
print('정확도 : ', acc) # 0.9415
print('정확도 : ', accuracy_score(test_y, pred)) # 0.9415
print('분류 보고서 : \n', classification_report(test_y, pred))











