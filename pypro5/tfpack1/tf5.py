# keras 모듈(라이브러리)을 사용하여 네트워크 구성
# 간단한 논리회로 분류 모델 

import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np 
from keras.optimizers import SGD, RMSprop, Adam
 
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])   # xor

model = Sequential()
# model.add(Dense(units=1, input_dim=2, activation='sigmoid'))
# model.add(unit=5, input_dim=2)
# model.Add(Activation('relu')) # Sigmoid와 tanh가 갖는 Grandient Vanishing 문제를 해결하기 위한 함수이다.
# Grandient Vanishing(기울기 소실) 문제는 Back Propagation에서 계산 결과와 정답과의 오차를 통해 가중치를 수정하는데,
# 입력층으로 갈수록 기울기가 작아져 가중치들이 업데이트 되지 않아 최적의 모델을 찾을 수 없는 문제입니다.
# model.add(unit=1)
# model.Add(Activation('sigmoid'))

model.add(Dense(units=5, input_dim=2, activation='relu'))
model.add(Dense(units=5, activation='relu')) 
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x,y,epochs=1000, batch_size=1, verbose=1)
loss_metrics = model.evaluate(x,y)
print(loss_metrics)

pred = (model.predict(x) > 0.5).astype('int32')
print('예측결과 : ', pred.flatten())

print(model.summary())
 
print()
print(model.input)
print(model.output)
print(model.weights)

print('**'*20)
print(history.history['loss'])
print(history.history['accuracy'])

# 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

import pandas as pd 
pd.DataFrame(history.history)['loss'].plot(figsize=(8,5))
plt.show()
























