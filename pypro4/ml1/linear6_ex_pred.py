# 검증이 끝났으니 이 상태로 배포
import joblib
import pandas as pd

mymodel = joblib.load('linear6m.model')

# 예측 : 새로운 Income, Advertising, Price, Age 값으로 Sales 추정
# x_new = pd.DataFrame({'Income':[80, 90, 60],'Advertising':[10, 5, 8],'Price':[100.0, 90.5, 85.0],'Age':[37, 45, 52]})
# new_pred = mymodel.predict(x_new)
# print('Sales 추정값 : ', new_pred.values)  # [9.72179008 9.38266604 9.39490659]   ***