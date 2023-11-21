# KNeighborsClassifier 클래스 사용 - iris dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus']=False
import joblib
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
# print(iris.DESCR)
print(iris.keys())
print(iris.feature_names)
print(iris.target_names)
print(iris.data[:2])
print(np.corrcoef(iris.data[:,2], iris.data[:,3])) #    0.96286

x = iris.data[:, [2, 3]]    # feature 'petal length (cm)', 'petal width (cm)'
y = iris.target
print(x[:2])
print(y[:2], set(y))    # {0, 1, 2}

print()
# train / test split (7 : 3) - 오버피팅 방지 기능
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0, shuffle=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,)

print()
'''
# scaling -  feature에 대해 data 표준화, 정규화 : 최적화 과정에서 안정성, 수련 속도를 향상, 오버피팅 or 언더피팅 방지 기능
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])
# scaling 값 원복
inverse_x_train = sc.inverse_transform(x_train)
print(inverse_x_train[:3])
'''
# 모델 작성 -----------------------------------------------------
model = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
model.fit(x_train, y_train)
print(model)
# ---------------------------------------------------------------

model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('총 갯수 : %d, 오류 수 : %d'%(len(y_test), (y_test !=y_pred).sum()))
print('분류 정확도 확인 1')
print('%.5f'%accuracy_score(y_test, y_pred))

print('분류 정확도 확인2')
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값'])
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))

print('분류 정확도 확인3 ')
print('test로 정확도는 ', model.score(x_test, y_test))
print('train 정확도는 ', model.score(x_train, y_train))

# 모델 성능이 만족스러운 경우 모델 저장
import joblib
joblib.dump(model, 'cla4model.sav') 

del model
mymodel = joblib.load('cla4model.sav')
print("새로운 값으로 분류 예측 -  'petal length (cm)', 'petal width (cm)' : 스케일링해서 학습했다면 예측 데이터도 스켈일링 함")
print(x_test[:2])
new_data = np.array([[5.1, 2.4], [0.1, 0.1], [5.6, 5.6], [8.1, 0.5]])
new_pred = mymodel.predict(new_data) # softmax함수가 제공한 결과에 대해 가장 큰 인덱스를 반환
print('예측 결과 : ', new_pred)
# print('soft 결과(날것) : ', mymodel.predict_proba(new_data))

# 시각화
def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')        # 점 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)       # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=x_combined_std, y=y_combined, classifier=mymodel, test_idx=range(105, 150), title='scikit-learn제공')   


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

 

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

 

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

 

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,

                                                    random_state=0)

 

# Learn to predict each class against the other
# OneVsOneClassifier 클래스를 사용하면 이진 클래스용 모형을 OvO 방법으로 다중 클래스용 모형으로 변환한다. 
# OneVsOneClassifier 클래스는 각 클래스가 얻는 조건부 확률값을 합한 값을 decision_function으로 출력한다.
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

 

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

 

# Compute micro-average ROC curve and ROC area
# 사이킷런 패키지는  roc_curve 명령을 제공한다. 
# 인수로는 타겟 y 벡터와 판별함수 벡터(혹은 확률 벡터)를 넣고 결과로는 변화되는 기준값과 그 기준값을 사용했을 때의 재현율을 반환한다.
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# AUC(Area Under the Curve)는 ROC curve의 면적을 뜻한다. 
# 위양성률(fall out)값이 같을 때 재현률값이 클거나 재현률값이 같을 때 위양성률값이 작을수록 AUC가 1에 가까운 값이고 좋은 모형이다.
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#Plot of a ROC curve for a specific class

 

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

 

# Plot ROC curves for the multiclass problem
# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))


# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

 

# Finally average it and compute AUC
mean_tpr /= n_classes

 

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

 

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

 

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

 

from itertools import cycle
# iterable에서 요소를 반환하고 각각의 복사본을 저장하는 반복자를 만든다. 반복 가능한 요소가 모두 소모되면 저장된 사본에서 요소를 리턴한다. 
# 반복 가능한 요소가 모두 소모될때까지 무한정 반복한다.
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))

 

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()









