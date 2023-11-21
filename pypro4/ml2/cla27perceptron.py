# 인공 신경망(Atificial Neural Network, ANN)은 사람의 뇌 속 뉴런의 작용을 본떠 패턴을 구성한 컴퓨팅 시스템의 일종.
# 퍼셉트론(Perceptron)은 가장 단순한 유형의 인공 신경망. 이런 유형의 네트워크는 대게 이진법 예측을 하는데 사용.
# 퍼셉트론은 데이터를 선형적(wx + b)으로 분리할 수 있는 경우에만 효과가 있습니다.

import numpy as np 
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
# label = np.array([0,0,0,1])     # and
# label = np.array([0,1,1,1])     # or
label = np.array([0,1,1,0])     # xor



ml = Perceptron(max_iter=10000, eta0=0.1, random_state=0).fit(feature, label) # max_iter:학습반복횟수, eta0:learning rate
print(ml)
pred = ml.predict(feature)
print('pred :', pred)
print('acc :', accuracy_score(label, pred))

print('\n다중 신경망 : MLP')
from sklearn.neural_network import MLPClassifier
ml2 = MLPClassifier(hidden_layer_sizes=(30), max_iter=10, solver='adam', learning_rate_init=0.01, verbose=1).fit(feature, label)
print(ml2)          # loss값이 점점 떨어지며 값을 찾아간다. # 연상 양을 늘리던가 학습 횟수를 증가하던가
pred2 = ml2.predict(feature)
print('pred2 :', pred2)
print('acc :', accuracy_score(label, pred2))


