# 실습 환경은 tensorflow 2.x   
# ex1) 단순선형회귀 - 경사하강법 함수 사용 1.x 

import tensorflow.compat.v1 as tf   # tensorflow 1.x 소스 실행 시
tf.disable_v2_behavior()            # tensorflow 1.x 소스 실행 시
import matplotlib.pyplot as plt

x_data = [1.,2.,3.,4.,5.]
y_data = [1.2,2.0,3.0,3.5,5.5]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = x * w + b
cost = tf.reduce_mean(tf.square(hypothesis - y))

print('\n경사하강법 메소드 사용------------')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()   # Launch the graph in a session.
sess.run(tf.global_variables_initializer())

w_val = []
cost_val = []

for i in range(501):
    _, curr_cost, curr_w, curr_b = sess.run([train, cost, w, b], {x:x_data, y:y_data})
    w_val.append(curr_w)
    cost_val.append(curr_cost)
    if i  % 10 == 0:
        print(str(i) + ' cost:' + str(curr_cost) + ' weight:' + str(curr_w) +' b:' + str(curr_b))

plt.plot(w_val, cost_val)
plt.xlabel('w')
plt.ylabel('cost')
plt.show()

print('--회귀분석 모델로 Y 값 예측------------------')
print(sess.run(hypothesis, feed_dict={x:[5]}))        # [5.0563836]
print(sess.run(hypothesis, feed_dict={x:[2.5]}))      # [2.5046895]
print(sess.run(hypothesis, feed_dict={x:[1.5, 3.3]})) # [1.4840119 3.3212316]


print('-----------------------')
# ex2) 단순선형회귀 - 경사하강법 함수 사용 2.x 
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np 

model = Sequential()
model.add(Dense(units=1, input_dim=1, activation='linear'))
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])     # mean_squared_error(평균제곱오차)
model.fit(x_data, y_data, batch_size=1, epochs=100, verbose=1)
print(model.evaluate(x_data, y_data))

pred = model.predict(x_data)
print('pred : ', pred.flatten())
 
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, pred, 'b')
plt.show()

# 결정계수
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, pred))

# 새로운 값으로 예측
new_x = [1.5,2.5,3.3]
print('새값 예측 결과 : ', model.predict(new_x).flatten())























