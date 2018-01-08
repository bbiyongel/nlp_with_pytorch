# 왜 강화학습을 써야 하나?

## Discriminative learning vs Generative learning

2012년 ImageNet competition에서 deeplearning을 활용한 AlexNet이 우승을 차지한 이래로 Computer Vision, Speech Recognition, Natural Language Processing 등을 차례로 deeplearning이 정복 해 왔습니다. Deeplearning이 뛰어난 능력을 보인 분야는 특히 classification 분야였습니다. 기존의 전통적인 방식과 달리 pattern recognition 분야에서는 압도적인 성능을 보여주었습니다. 이러한 classification 문제는 보통 ***Discriminative Learning***에 속하는데 이를 일반화 하면 다음과 같습니다.

$$
\hat{y} = argmax_{y \in Y} P(y|x)
$$

주어진 $$x$$에 대해서 최대의 확률 값을 갖는 $$y$$를 찾아내는 것 입니다. 이러한 조건부 확률 모델을 학습하는 것을 discriminative learning이라고 부릅니다. 하지만 이에 반해 ***Generative Learning***은 확률 $$ P(x) $$ 자체를 modeling 하는 것을 이릅니다. Generative learning이 훨씬 더 어려운 task에 속하게 됩니다. 예를 들어 

1. 사람의 생김새($$x$$)가 그림으로 주어졌을 때 성별($$y$$)를 예측하는 것
1. 사람의 생김새($$x$$) 자체를 예측하는(or 그림으로 그려내는) 것

두가지 case를 비교하면 2번째 case가 훨씬 더 난이도가 높음을 쉽게 알 수 있습니다. 요새에는 pattern recognition을 비롯한 discriminative learning은 이제 deeplearning으로 당연하게 잘 해결되기 때문에 사람들의 관심과 연구 트렌드는 위와 같은 generative learning으로 집중되고 있습니다.

## Generative Adversarial Network (GAN)

2016년부터 주목받기 시작하여 2017년에 가장 큰 화제였던 분야는 단연 GAN이라고 말할 수 있습니다. Variational Auto Encoder(VAE)와 함께 Generative learning을 대표하는 방법 중에 하나입니다. GAN을 통해서 우리는 사실같은 이미지를 생성해내고 합성해내는 일들을 딥러닝을 통해 할 수 있게 되었습니다. 이러한 합성/생성 된 이미지들을 통해 자율주행과 같은 실생활에 중요하고 어렵지만 데이터셋을 얻기 힘든 task들을 해결 하는데 큰 도움을 얻을 수 있으리라고 예상 됩니다. (실제로 GTA게임을 통해 자율주행을 훈련하려는 시도는 이미 유명합니다.)

## GAN과 NLP

## 강화학습 소개
