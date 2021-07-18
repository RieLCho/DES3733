# DES3733

개별연구(AI모델 보안 강화 방법 연구)

[https://github.com/Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

IBM사에서 오픈소스로 개발을 진행하고 있는 Adversarial Robustness Toolbox(이하 ART)를 사용한 AI 모델 보안 강화에 관한 포스팅입니다. 다른 학우분들께 도움이 되고 싶어 이렇게 글을 남겨봅니다.

### 개요

최근 AI 모델을 해킹하여 악용하기 위한 다양한 사례들이 발생하고 있습니다. 안전하게 AI 모델을 사용할 수 있도록 AI 모델 생성을 위한 데이터 수집, 학습, 평가, 활용시 발생할 수 있는 보안 취약점을 분석하고 이를 사전에 제거, 방어, 검출 하기 위한 기법을 연구합니다.

#### 개요 안에 답이 있다

처음 해당 연구주제를 접했을 때는 정말 막막했습니다. 아무리 구글링을 해보고, GitHub을 뒤져봐도 사람들이 ART를 사용한 흔적이 전혀 없었기 때문입니다. 하지만 지금와서 다시 연구 개요를 읽어보니 교수님께서 원하시는 바를 정확히 이해할 수 있게 되었습니다. 

**데이터 수집, 학습, 평가, 활용시 발생할 수 있는 보안 취약점을 분석하고**

\--> Adversarial Attack이 어떤 방식으로 행해지는지를 분석하고

**사전에 제거, 방어, 검출 하기 위한 기법**

\--> Adversarial Attack을 제거 (preprocessor/postprocessor), 방어 (adversarial trainer), 검출 (detector) 기법을 연구

### 시작은 무조건 공식문서부터

[https://youtu.be/hWwJF-fU2Lg](https://youtu.be/hWwJF-fU2Lg)

<iframe src="https://www.youtube.com/embed/hWwJF-fU2Lg" width="480" height="360" frameborder="0" allowfullscreen="true"></iframe>

시간되시면 영상 한번 봐주세요

제시된 ART(adversarial robustness toolbox)를 가장 잘 아는 사람은 누구일까요? 바로 해당 라이브러리를 작성한 사람들입니다. ART의 경우는 문서화가 참 잘 되어 있는 편이라, 무조건 RTFM하세요.

[https://adversarial-robustness-toolbox.readthedocs.io/en/latest/](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/)

[https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/README.md](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/README.md)

사실 위의 예제만 돌려도 반은 먹고 들어가는데, 읽어보면 상당히 여러가지 데이터들을 제공하고 있습니다. 주어진 AI 모델만 응용해도 될 정도로 말이죠.

### AI 모델 보안 강화 방법?

만약 제가 ART를 사용하지 않았다면 다음과 같은 방식을 선택했을 것입니다.

텐서플로우의 경우 계속하여 익스플로잇을 발견하고 해당 익스플로잇을 패치하여 버전 업을 하겠죠. 따라서 예전 버전의 텐서플로우를 사용하여 이미 밝혀진 익스플로잇을 공격하고, 해당 패치 내역을 따라가서 어떤 식으로 코드를 수정했는지 확인하여 이를 분석했을 것입니다.

하지만 우리는 ART를 사용해야하는 운명이기 때문에, 다음과 같은 플로우를 따라야 합니다.

1\. 데이터셋을 학습시키고 해당 모델의 정확도를 측정한다

2\. 해당 모델에 Adversarial Attack을 한 뒤 정확도를 측정한다

3\. \* 이를 방어할 방법을 찾고, 해당 방어법을 사용한다 \*

4\. 방어한 모델에 똑같이 Adversarial Attack을 행한뒤 정확도를 측정한다

### 예시

MNIST 데이터 셋을 기반으로 만든 AI 모델에 FGSM (Fast Gradient Signed Method)공격을 행한뒤, Adversarial Trainer를 통해 새로 학습시킨 모델과의 정확도를 비교해보며 공격 방어에 성공하였는지 확인해봅니다.

```
# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
```

MNIST 데이터셋을 각각 x\_train - y\_train 그리고 x\_test - y\_test에 불러옵니다. x에 해당 데이터, y에 라벨 정보가 저장됩니다.

```
# Step 2: Create the model
model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(
    loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]
)
```

이후 모델을 생성합니다. 해당 모델은 ART 예제에서 가져왔습니다.

```
# Step 3: Create the ART classifier
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
```

classifier를 정의합니다

```
# Step 4: Train the ART classifier
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=20)
```

해당 classifier를 학습시킵니다.

```
# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("정상적으로 학습시킨 MNIST 모델의 정확도: {}%".format(accuracy * 100))
```

이후 해당 classifier의 predict를 통해 정상적으로 학습시킨 정확도를 측정합니다.

```
# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)
```

FGSM을 통해 x\_test\_adv라는 공격당한 테스트 셋을 생성합니다. eps 값을 조정하여 공격 강도를 수정할 수 있습니다.

```
# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("MNIST에 FGSM 공격을 가한 후 정확도: {}%".format(accuracy * 100))
```

이제 아까 생성했던 classifier에 x\_test\_adv를 predict 하게끔하여 일반적으로 학습시킨 모델의 공격당한 테스트 셋의 정확도를 측정합니다.

```
# Step 8: Train with AdversarialTrainer for accuracy comparision
# Paper link: https://arxiv.org/abs/1705.07204
AdversarialTrainer(classifier=classifier, attacks=attack, ratio=0.5).fit(x=x_train, y=y_train, batch_size=64, nb_epochs=20)
```

이제 새로 Adversarial Trainer를 사용하여 evasion 공격을 방어할 수 있는 모델을 새로 생성하고 학습시킵니다.

```
# Step 9: Evaluate the ART classifier on adversarial test examples
predictions = AdversarialTrainer(classifier=classifier, attacks=attack, ratio=0.5).predict(x=x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("AdversarialTrainer - FGSM을 방어한 후의 모델 정확도: {}%".format(accuracy * 100))
```

Adversarial Trainer를 통해 새로 만든 모델에 아까 공격시켜 만든 x\_test\_adv를 predict 시켜 해당 모델을 공격한 정확도를 확인합니다.

![image](https://user-images.githubusercontent.com/13748138/126075035-da48703a-90af-4488-a795-8fecb601ee70.png)

이를 통해 Adversarial Trainer가 실제로 효과를 보임을 증명했습니다. 

이후 분석 및 평가는 여러분의 몫입니다. 여러가지 변수를 줘보며 데이터를 추출해보세요.

### TIP

\- 텐서플로우를 사용하실 때는 v2가 아닌 v1을 사용하는 것이 속 편합니다. (가끔 v2를 지원하는 함수들도 있긴한데, v1용이 많아요)

\- 모르는게 있으면 무조건 공식 문서 살펴보기. 꼭 문서 안열어봐도 함수 안에 까보면 다 주석으로 설명되어 있습니다.

![image](https://user-images.githubusercontent.com/13748138/126075044-0c786578-f782-4005-a65a-f516b228378c.png)

\- 아무래도 학습시키고 여러가지 실험해보는데 시간이 많이 필요하니 넉넉하게 시간을 가지고 보고서를 작성하는 것이 정신건강에 좋습니다.

### 참고하면 분명히 도움이 될 문헌들

\- 이재원, 김종효, 「딥러닝 기반 컴퓨터 보조진단의 취약성: 폐 결절 검출 모델에 대한 실험적 적대적 공격」, 대한의학영상정보학회지 2018;24:1-10

\- Goodfellow IJ, Jonathon S, Christian S. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014

\- Moosavi-Dezfooli SM, Alhussein F, Pascal F. Deepfool: a simple and accurate method to fool deep neural networks. Proceedinigs of the IEEE Conference on Computer Vision and Pattern Recognition. 2016; 2574-

\- Jang U, Xi W, Somesh J. Objective Metrics and Gradient Descent Algorithms for Adversarial Examples in Machine Learning. Proceedings of the 33rd  Annual Computer Security Applications Conference. ACM, 2017; 262-277

\- Szegedy C, et al. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.

\- Florian Tramèr Ensemble Adversarial Training: Attacks and Defenses, arXiv:1705.07204, 2017.

\- Towards Deep Learning Models Resistant to Adversarial Attacks, arXiv:1706.06083, 2019.
