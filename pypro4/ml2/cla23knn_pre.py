from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import mglearn    # pip install maglean
import matplotlib.pyplot as plt 
plt.rc('font', family='malgun gothic')

# 가장 간단한 k-NN 알고리즘은 가장 가까운 훈련 데이터 포인트 하나를 최근
# 단순히 이 훈련 데이터 포인트의 출력이 예측이 된다.
#---------------------------
# Classification
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()      # 과적합

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=9)
plt.show()      # 과소적합

# Regression


