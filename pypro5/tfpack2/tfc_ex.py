import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 데이터 로드
data = pd.read_csv('../testdata/HR_comma_sep.csv')

# 범주형 변수를 원핫 인코딩으로 인코딩
data_encoded = pd.get_dummies(data, columns=['sales'], drop_first=True)

# 특성 (X) 및 목표 변수 (y)로 분할
X = data_encoded.drop('salary', axis=1)
y = data_encoded['salary']

# 훈련-테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Random Forest 모델
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 예측
rf_predictions = rf_model.predict(X_test)

# 정확도
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest 정확도: {:.3f}".format(rf_accuracy))


#Keras 모델
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

# 목표 변수 인코딩
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 특성 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keras 모델
keras_model = Sequential()
keras_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
keras_model.add(Dense(32, activation='relu'))
keras_model.add(Dense(len(label_encoder.classes_), activation='softmax'))

keras_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
keras_model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# 모델 평가
keras_loss, keras_accuracy = keras_model.evaluate(X_test_scaled, y_test_encoded)
print("Keras 정확도: {:.3f}".format(keras_accuracy))

#분류정확도 비교?
print("Random Forest 정확도: {:.3f}".format(rf_accuracy))
print("Keras 정확도: {:.3f}".format(keras_accuracy))

