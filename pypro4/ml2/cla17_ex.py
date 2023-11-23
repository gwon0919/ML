'''
Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
dataset에 대해 학습을 위한 train과 test로 구분하고 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
임의의 값을 넣어 분류 결과를 확인하시오.     

feature 칼럼 : 문자 데이터 칼럼은 제외
label 칼럼 : AHD(중증 심장질환)
'''
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

'''

data = pd.read_csv("../testdata/Heart.csv")

# print(data.dtypes)
print(data.isnull().sum())

feature = data.drop(['AHD','ChestPain','Thal','Unnamed: 0'], axis=1)
print(feature.head(5))
label = data.AHD
print(label.head(3))

# train / test
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=158)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# 정규화
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# model 생성
# model = svm.SVC().fit(x_train_scaled, y_train)
# model = svm.SVC().fit(x_train, y_train)


model = svm.SVC().fit(x_train, y_train)


# pred = model.predict(x_test_scaled)
pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)


ac_score = metrics.accuracy_score(y_test, pred)
print('분류 정확도 : ', ac_score)


print(feature.columns)
# 새로운 값으로 분류 예측
newdata = pd.DataFrame({'Age':[69, 59, 70],'Sex':[1,0,1],'RestBP':[145,120,160],'Chol':[240,250,280],'Fbs':[1,0,0]
                           ,'RestECG':[2,2,2],'MaxHR':[150,108,129],'ExAng':[0,1,1], 'Oldpeak':[2.5,1.5,2.6], 'Slope':[3,2,2], 'Ca':[0.0,3.0,2.0]})
newPred = model.predict(newdata)
print('newPred : ',newPred)



sns.pairplot(data[['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'AHD']],
             hue='AHD', markers=['o', 's'])
plt.show()
'''

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드 및 전처리
data = pd.read_csv('../testdata/titanic_data.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age', 'Fare'])
data.loc[data["Sex"] == "male", "Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1

# 특성과 레이블 선택
features = data[["Pclass", "Sex", "Fare"]]
label = data["Survived"]

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=12)

# 모델 생성
clf = DecisionTreeClassifier(random_state=12)

# 모델 학습
clf.fit(x_train, y_train)

# 예측
y_pred = clf.predict(x_test)

# 정확도 평가
acc = accuracy_score(y_test, y_pred)
print("모델 정확도: ", acc)

















