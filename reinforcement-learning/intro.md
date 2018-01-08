# 왜 강화학습을 써야 하나?

## Discriminative learning vs Generative learning

2012년 ImageNet competition에서 deeplearning을 활용한 AlexNet이 우승을 차지한 이래로 Computer Vision, Speech Recognition, Natural Language Processing 등을 차례로 deeplearning이 정복 해 왔습니다. Deeplearning이 뛰어난 능력을 보인 분야는 특히 classification 분야였습니다. 기존의 전통적인 방식과 달리 pattern recognition 분야에서는 압도적인 성능을 보여주었습니다. 이러한 classification 문제는 보통 ***Discriminative Learning***에 속하는데 이를 일반화 하면 다음과 같습니다.

$$
\hat{y} = argmax_{y \in Y} P(y|x)
$$

주어진 $$x$$에 대해서 최대의 확률 값을 갖는 $$y$$를 찾아내는 것 입니다. 이러한 조건부 확률 모델을 학습하는 것을 discriminative learning이라고 부릅니다. 하지만 이에 반해 ***Generative Learning***은 확률 $$ P(x) $$ 자체를 modeling 하는 것을 이릅니다. Generative learning이 훨씬 더 어려운 task에 속하게 됩니다. 예를 들어 

1. 사람의 생김새($$x$$)가 그림으로 주어졌을 때 성별($$y$$)를 예측하는 것
1. 사람의 생김새($$x$$)와 성별($$y$$) 자체를 예측하는(or 그림으로 그려내는) 것

두가지 case를 비교하면 2번째 case가 훨씬 더 난이도가 높음을 쉽게 알 수 있습니다. 그리고 이것을 수식으로 일반화 하면 아래와 같습니다.

$$
P(x, y)
$$

요새는 pattern recognition을 비롯한 discriminative learning은 이제 deeplearning으로 당연하게 잘 해결되기 때문에 사람들의 관심과 연구 트렌드는 위와 같은 generative learning으로 집중되고 있습니다.

## Generative Adversarial Network (GAN)

2016년부터 주목받기 시작하여 2017년에 가장 큰 화제였던 분야는 단연 GAN이라고 말할 수 있습니다. Variational Auto Encoder(VAE)와 함께 Generative learning을 대표하는 방법 중에 하나입니다. GAN을 통해서 우리는 사실같은 이미지를 생성해내고 합성해내는 일들을 딥러닝을 통해 할 수 있게 되었습니다. 이러한 합성/생성 된 이미지들을 통해 자율주행과 같은 실생활에 중요하고 어렵지만 데이터셋을 얻기 힘든 task들을 해결 하는데 큰 도움을 얻을 수 있으리라고 예상 됩니다. (실제로 GTA게임을 통해 자율주행을 훈련하려는 시도는 이미 유명합니다.)

![](https://sthalles.github.io/assets/dcgan/GANs.png)
Generative Adversarial Network overview - Image from [web](https://sthalles.github.io/intro-to-gans/)

위와 같이 ***Generator($$G$$)***와 ***Discriminator($$D$$)*** 2개의 모델을 각기 다른 목표를 가지고 동시에 훈련시키는 것입니다. $$D$$는 임의의 이미지를 입력으로 받아 이것이 실제 존재하는 이미지인지, 아니면 합성된 이미지인지 탐지 해 내는 역할을 합니다. $$G$$는 어떤 이미지를 생성 해 내되, $$D$$를 속이는 이미지를 만들어 내는 것이 목표입니다. 이렇게 두 모델이 잘 균형을 이루며 $$ minmax $$ 게임을 펼치게 되면, $$G$$는 결국 훌륭한 이미지를 합성 해 내는 Generator가 됩니다.

여기에서는 GAN을 자세히 설명하지 않고 넘어가도록 하겠습니다.

## GAN과 NLP

위와 같이 GAN은 Computer Vision(CV)분야에서 대성공을 이루었지만 NLP에서는 적용이 어려웠습니다. 그 이유는 Natural Language 자체의 특성에 있습니다. 이미지라는 것은 어떠한 continuous한 값들로 채워진 2차원의 matrix입니다. 하지만 이와 달리 단어라는 것은 descrete한 symbol로써, 언어라는 것은 어떠한 descrete한 값들의 sequential한 배열 입니다. 비록 우리는 **NNLM**이나 **NMT Decoder**를 통해서 latent variable로써 언어의 확률을 모델링 $$ P(w_1,w_2,\cdots,w_n)$$ 하고 있지만, 결국 언어를 나타내기 위해서는 해당 확률 모델에서 ***sampling***(또는 argmax)을 하는 stochastic한 과정을 거쳐야 합니다.

$$
\hat{w}_t = argmax P(w_t|w_1,\cdots,w_{t-1})
$$

Sampling 또는 argmax의 연산은 gradient를 전달 할 수 없는 stochastic 연산입니다. -- 오직 deterministic한 연산만 gradient가 back-propagation 가능합니다. 이러한 이유 때문에 보통은 GAN을 NLP에는 적용할 수 없는 인식이 지배적이었습니다. 하지만 강화학습을 사용함으로써 Adversarial learning을 NLP에도 적용할 수 있게 되었습니다.

## 강화학습 소개

위와 같이 GAN을 사용하기 위함 뿐만이 아니라, 강화학습은 매우 중요합니다. 우