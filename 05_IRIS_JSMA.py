# 2019112130 Yangjin Cho
# 05_IRIS_JSMA_DEFEND
# Independant Capstone AI Model Security
import keras
import time
import datetime
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

from art.attacks.evasion import SaliencyMapMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_iris
from art.defences.trainer import AdversarialTrainer

start = time.time()
# Step 1: Load the IRIS dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_iris()

# Step 2: Create the model
# https://www.kaggle.com/rushabhwadkar/deep-learning-with-keras-on-iris-dataset
model = Sequential()
model.add(Dense(10,input_dim=4, activation='relu',kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))
model.add(Dense(7, activation='relu', kernel_initializer='he_normal',kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))
model.add(Dense(5, activation='relu', kernel_initializer='he_normal',kernel_regularizer=keras.regularizers.l1_l2(l1=0.001,l2=0.001)))
model.add(Dense(3,activation='softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
)
# Step 3: Create the ART classifier
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

# Step 4: Train the ART classifier
classifier.fit(x_train, y_train, batch_size=7, nb_epochs=700)

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("정상적으로 학습시킨 IRIS 모델의 정확도: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = SaliencyMapMethod(classifier=classifier)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("IRIS에 JSMA 공격을 가한 후 정확도: {}%".format(accuracy * 100))

sec = time.time() - start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print(times)