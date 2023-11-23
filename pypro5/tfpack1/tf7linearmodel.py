# 단순 선형 모델 작성
# 1) keras의 내장 api  사용 - Sequential : 다음번 예제
# 2) GradientTape 객체를 이용해 모델을 구현 - 유연하게 복잡한 로직을 처리할 수 있다. 
# TensorFlow는 GradientTape을 이용하여 쉽게 오차 역전파를 수행할 수 있다. 

import tensorflow as tf 
import numpy as np 
from keras.optimizers import SGD, RMSprop, Adam
tf.random.set_seed(2)

w = tf.Variable(tf.random.normal((1,)))
b = tf.Variable(tf.random.normal((1,)))
print(w.numpy(), b.numpy())
opti= SGD()

@tf.function
def trainModel(x, y):
    with tf.GradientTape() as tape:
        hypo = tf.add(tf.multiply(w,x), b)   # wx + b
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))  # cost function
    grad = tape.gradient(loss, [w, b])        # 자동 미분 (loss를 w와 b로 미분)
    opti.apply_gradients(zip(grad, [w, b]))
    return loss 
    
x = [1.,2.,3.,4.,5.]
y = [1.2,2.0,3.0,3.5,5.5]
print(np.corrcoef(x,y)) # 0.9749


w_val = []
cost_val = []

for i in range(1,101):
    loss_val = trainModel(x, y)
    cost_val.append(loss_val.numpy())
    w_val.append(w.numpy())
    if i % 10 == 0:
        print(loss_val)
        
print('cost_val :', cost_val)
print('w_val : ', w_val)

import matplotlib.pyplot as plt
plt.plot(w_val, cost_val, 'o')
plt.xlabel('w')
plt.ylabel('cost')
plt.show()
    

print('cost가 최소일 떄 w:', w.numpy())
print('cost가 최소일 떄 b:', b.numpy())

y_pred = tf.multiply(x, w) + b   # 선형회귀식 완성
print('예측값 : ', y_pred.numpy())

plt.plot(x,y, 'ro', label='real y')
plt.plot(x,y_pred, 'b-', label='pred')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 새 값으로 예측 
new_x = [3.5, 9.0]
new_pred = tf.multiply(new_x, w) + b 
print('예측 결과 ', new_pred.numpy())   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








