# 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from dask.dataframe.shuffle import shuffle
from sklearn.metrics import accuracy_score

data = {
    '요일': ['토', '토', '토', '화', '토', '월', '토', '토', '토', '토', '토', '토', '토', '토', '일', '월', '화', '수', '목', '금', '토', '토', '토', '토', '일', '토', '일', '토'],
    '외식유무': [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    '소득수준': [57, 39, 28, 60, 31, 42, 54, 65, 45, 37, 98, 60, 41, 52, 75, 45, 46, 39, 70, 44, 74, 65, 46, 39, 60, 44, 30, 34]
}

df = pd.DataFrame(data)
print(df.head(), df.shape)
# Convert '외식유무' to numeric (0 and 1)
df['외식유무'] = pd.to_numeric(df['외식유무'])

# Adjusted formula
formula = '외식유무 ~ 소득수준 + C(요일)'

model = smf.glm(formula=formula, data=df, family = sm.families.Binomial()).fit()
print(model)
print(model.summary())

income_level = int(input("소득 수준을 입력하세요: "))

input_data = {'소득수준': [income_level]}
input_df = pd.DataFrame(input_data)
input_df = pd.get_dummies(input_df, columns=['요일'], drop_first=True)

prediction = model.predict(input_df)[0]

# 결과 출력
if prediction == 1:
    print("외식할 가능성이 있습니다.")
else:
    print("외식할 가능성이 낮습니다.")