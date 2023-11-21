import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic')


# 상관관계 문제)
# Advertising.csv 파일을 읽어 tv,radio,newspaper 간의 상관관계를 파악하시오. 
# 그리고 이들의 관계를 heatmap 그래프로 표현하시오. 
data = pd.read_csv( '../testdata/Advertising.csv')

print(data.head())

df = pd.DataFrame(data, columns=('tv','radio','newspaper'))
print(df)

# heatmap 그래프 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

     

