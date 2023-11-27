# 동물을 7가지 type으로 분류 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

dataset = np.loadtxt("../testdata/zoo.csv", delimiter=',')
print(dataset[0],dataset.shape) # (101, 17)
print(set(dataset[:, -1]))  # {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0} 다항 분류 대상

x_data = dataset[:, 0:-1]  # feature
y_data = dataset[:, -1]     # label

print(x_data[:3])
print(y_data[:3])

# label에 원핫 처리 안함

model = Sequential()
model.add(Dense(32, input_shape=(16,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))
print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit(x_data, y_data, batch_size=10, epochs=100, verbose=0, validation_split=0.3)
print(model.evaluate(x_data, y_data, batch_size=10, verbose=0))

# 시각화 
history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(loss, 'b-', label='loss')
plt.plot(val_loss, 'r--', label='val_loss')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.plot(loss, 'b-', label='accuracy')
plt.plot(val_loss, 'r--', label='val_accuracy')
plt.ylabel('accuracy')
plt.legend()
plt.show()

pred_datas = x_data[:5]
preds = [np.argmax(i) for i in model.predict(pred_datas)]
print('예측 값: ', preds)
print('실제 값: ', y_data[:5].flatten())




























