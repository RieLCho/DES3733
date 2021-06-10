# 2019112130 Yangjin Cho
# 04_CIFAR10_DEEPFOOL_DEFEND
# Independant Capstone AI Model Security
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np

from art.attacks.evasion import DeepFool
from art.estimators.classification import KerasClassifier
from art.utils import load_cifar10
from art.defences.trainer import AdversarialTrainer

# Step 1: Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

# Step 2: Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Step 3: Create the ART classifier
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

# Step 4: Train the ART classifier
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=10)

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("정상적으로 학습시킨 CIFAR-10 모델의 정확도: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = DeepFool(classifier=classifier, max_iter=5, epsilon=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 7: AdversarialTrainer
# Paper link: https://arxiv.org/abs/1705.07204
AdversarialTrainer(classifier=classifier, attacks=attack, ratio=0.5).fit(x=x_train, y=y_train, batch_size=64, nb_epochs=10)

# Step 8: Evaluate the ART classifier on adversarial test examples
predictions = AdversarialTrainer(classifier=classifier, attacks=attack, ratio=0.5).predict(x=x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("AdversarialTrainer - DeepFool을 방어한 후의 모델 정확도: {}%".format(accuracy * 100))