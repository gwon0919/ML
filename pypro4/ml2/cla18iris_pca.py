# 주성분 분석(Principal Component Analysis, PCA)은 가장 널리 사용되는 차원 축소 기법(특성 공학) 중 하나로,
# 원 데이터의 분포를 최대한 보존하면서 고차원 공간의 데이터들을 저차원 공간으로 변환한다.
# PCA는 기존의 변수를 조합(국어 + 영어 -> 어학) 하여 서로 연관성이 없는 새로운 변수, 즉 주성분들을 만들어 낸다.
# n개의 관측치와 p개의 변수로 구성된 데이터를 상관관계가 없는 k개의 변수로 구성된 데이터로 요약하는 방식
# 이 때 요약된 변수는 기존 변수의 선형 조합으로 원래 성질을 최대한 유지해야 한다.
# 선형대수 관점으로 볼 때 입력 데이터의 공분산 행렬을 고유값 분해하고 이렇게 구한 고유벡터에 입력데이터를 선형 변환하는 것이다.
# 이 고유벡터가 PCA의 주성분 벡터로서 입력 데이터의 분산이 큰 방향을 나타낸다.

# iris dataset을 사용
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from sklearn.datasets import load_iris
from ml1.linear8 import feature
plt.rc('font', family='malgun gothic')

iris = load_iris()
# print(iris.feature_names)
n = 10      # 관찰값 
x = iris.data[:n, :2]       # sepal - length, width
# print('차원 축소 전 x : ', x, x.shape, type(x))
# print(x.T)

# 시각화
# plt.plot(x.T, 'o:')
# plt.grid()
# plt.title('iris 크기 특성')
# plt.xlabel('특성 종류')
# plt.ylabel('특성값')
# plt.legend(["표본{}".format(i + 1) for i in range(n)])
# plt.show()

# scatter plot 
df = pd.DataFrame(x)
print(df.head(2))
# ax = sns.scatterplot(x=df[0], y=df[1], data = pd.DataFrame(x), markers='s', s=100, color=['b'])
# for i in range(10):
#     ax.text(x[i, 0]-0.05, x[i, 1] + 0.03, '표본{}'.format(i+1))
# plt.xlabel('꽃받침 길이')
# plt.ylabel('꽃받침 너비')
# plt.axis('equal')
# plt.show()

print()
print(x.shape) # (10, 2)
# PCA로 차원 축소 (꽃받침 길이, 꽃받침 너비 -> 한 개의 값)

pca1 = PCA(n_components=1)
x_low = pca1.fit_transform(x)
print('x_low : ', x_low[:2], ' ', x_low.shape)

x2 = pca1.inverse_transform(x_low)      # 주성분 분석된 값을 원복
print('차원 복귀 후 x2 : ', x2, ' ', x2.shape) # 5.1  3.5 ->  0.30270263 -> 5.06676112 3.53108532

# scatter plot 
'''
df = pd.DataFrame(x)
ax = sns.scatterplot(x=df[0], y=df[1], data = pd.DataFrame(x), markers='s', s=100, color=['r'])
for i in range(n):
    d = 0.03 if x[i,1] > x2[i,1] else -0.04
    ax.text(x[i, 0]-0.065, x[i, 1] + d, '표본{}'.format(i+1))
    plt.plot([x[i,0], x2[i,0]], [x[i,1], x2[i,1]], 'k--')
plt.plot(x2[:,0], x2[:,1], 'o-', color='b', markersize=10)
plt.plot(x[:,0].mean(), x[:,1].mean(), 'D', color='y', markersize=10)
plt.axvline(x[:,0].mean(), c='gray')
plt.axhline(x[:,1].mean(), c='gray')
plt.xlabel('꽃받침 길이')
plt.ylabel('꽃받침 너비')
plt.axis('equal')
plt.show()
'''

print('4개의 열을 2개로 축소')
x = iris.data
print('원래 x : ', x[:3], ' ', x.shape)
pca2 = PCA(n_components=2)
x_low2 = pca2.fit_transform(x)      # 비지도 학습이므로 y(target)는 주어지지 않는다.
print('x_low2 : ', x_low2[:2], ' ', x_low2.shape)

x4 = pca2.inverse_transform(x_low2)      # 주성분 분석된 값을 원복
print('차원 복귀 후 x4 : ', x4[:3], ' ', x4.shape)   # 근사행렬로 변환

# [[5.1 3.5 1.4 0.2] [4.9 3.  1.4 0.2] [4.7 3.2 1.3 0.2]]   (150, 4)

# [[-2.68412563  0.31939725] [-2.71414169 -0.17700123]]     (150, 2)

# [[5.08303897 3.51741393 1.40321372 0.21353169] [4.7462619  3.15749994 1.46356177 0.24024592] [4.70411871 3.1956816  1.30821697 0.17518015]]  (150, 4)

print(pca2.explained_variance_ratio_)   # [0.92461872 0.05306648] 변동성 비율

print()
# 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
iris1 = pd.DataFrame(x, columns=['sepal lengt', 'sepal width', 'petal length', 'petal width'])
print(iris1.head(3))
iris2 = pd.DataFrame(x_low2, columns=['var1','var2'])
print(iris2.head(3))

from sklearn import svm, metrics
print('-------------')
feature1 = iris1.values
print(feature1[:3])
label = iris.target
print(label[:3])

model1 = svm.SVC(C=0.1, random_state=0).fit(feature1,label)
pred1 = model1.predict(feature1)
print('model1 acc : ', metrics.accuracy_score(label,pred1))

feature2 = iris2.values
print(feature1[:3])
print(label[:3])

print('PCA 데이터로 처리')
model2 = svm.SVC(C=0.1, random_state=0).fit(feature2,label)
pred2 = model2.predict(feature2)
print('model2 acc : ', metrics.accuracy_score(label,pred2))








