# keras 모듈(라이브러리)을 사용하여 네트워크 구성
# 간단한 논리회로 분류 모델 

import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np 
from keras.optimizers import SGD, RMSprop, Adam

# 1. 데이터 수집 및 가공 
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

# 2. 모델 구성(설정)
# model = Sequential([
#     Dense(input_dim=2, units=1),
#     Activation('sigmoid')
#     ])
model = Sequential()
# model.add(Dense(units=1, input_dim=2))
# model.add(Activation('sigmoid'))
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))

# 3. 모델 학습 과정 설정(컴파일)
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc']) # metrics =>성능지표 확인을 위함 # regression-> mse
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy']) # momentum은 적당히 큰 수부터 작게 내린다.

# optimizer : 입력데이터와 손실함수를 업데이트하는 메커니즘이다. 손실함수의 최소값을 찾는 알고리즘을 옵티마이져라 한다. 
# 옵티마이저를 통해 코스트를 미니마이즈화 # sgd -> rmsprop -> adam 으로 실행 해본다. 

# 4. 모델 학습시키기 : 더 나은 표현을 찾는(w를 갱신) 자동화 과정 
model.fit(x, y, epochs=10, batch_size=1, verbose=0)
# batch_size : 훈련데이터를 여러 개의 작은 묶음(batch)으로 만들어 가중치(w)를 갱신. 1 epoch 시 사용하는 dataset의 크기  

# 5. 모델 평가 (test) 
loss_metrics = model.evaluate(x, y, batch_size=1, verbose=0)
print(loss_metrics)  # [0.3983510434627533, 0.75]

# 6. 학습결과 확인 : 예측값 출력
# pred = model.predict(x, batch_size=1, verbose=0)
pred = (model.predict(x) > 0.5).astype('int32')
print('예측결과 : ', pred.flatten()) # 차원 떨어뜨리기

# 7. 모델 저장 
model.save('tf4model.h5')   # hdf5 

# 8. 모델 읽기
from keras.models import load_model
mymodel = load_model('tf4model.h5')

mypred = (mymodel.predict(x) > 0.5).astype('int32')
print('예측결과 : ', mypred.flatten())
























