# cost와 accuracy/r2core 는 반비계한다.
import numpy as np 
import math
from xgboost.testing import hypothesis

# 모델 학습이 끝나고 예측 값 구하기 단계(가정)
real = np.array([10,3,3,2,11])
# pred = np.array([11,5,2,4,8])  # 실제 값과 차이가 적음 3.8
pred = np.array([15,8,2,9,2])    # 실제 값과 차이가 큼 36.2
cost = 0
for i in range(5):
    cost += math.pow(pred[i] - real[i], 2)
    print(cost)
    
print(cost / len(pred))

print('-----------------------')
import tensorflow as tf 
import matplotlib.pyplot as plt 

x = [1.,2.,3.,4.,5.]
y = [2.,4.,6.,8.,10.]
b = 0
# hypothesis = w * x + b linear model을 구하는 수식
# cost = tf.reduce_sum(tf.pow(hypothesis - y, 2)) / len(x)
# cost = tf.reduce_mean(tf.pow(hypothesis - y, 2))

w_val =[]
cost_val = []

for i in range(-30,50):
    feed_w = i * 0.1
    hypothesis = tf.multiply(feed_w, x) + b  # wx + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    cost_val.append(cost)
    w_val.append(feed_w)
    print(str(i) + '번 수행' + ', cost: ' + str(cost.numpy()) + ', weight: ' + str(feed_w))
    
plt.plot(w_val, cost_val)
plt.xlabel('w')
plt.ylabel('cost')
plt.show()    













